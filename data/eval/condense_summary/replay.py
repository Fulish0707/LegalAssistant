"""Condenser history check, replay layer (PROTOCOL.md, main judgment).

Calls only the condenser, under both arms, on identical inputs:
  - every condenser call of the recorded base run that reached the model (its message and the
    stored history, as run.py logged them), and
  - the hand-written cases in referent_cases.json (one user turn, one answer turn).
Arm base swaps legal_rag.rag.condenser.condenser_view for the identity, which is the preregistration
commit's behaviour; arm fix uses the code as committed. Each arm runs every input twice.

Usage (serving .env loaded, cwd = repo root, <tree> = worktree at the fix commit):
  ( set -a; . ./.env; set +a; exec env -u PYTHONHOME PYTHONPATH=<tree>/src .venv/bin/python \\
      data/eval/condense_summary/replay.py <recorded base run .jsonl> )
Writes runs/<tag>-replay.jsonl. No conversation is stored.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import legal_rag
import legal_rag.rag.condenser as condenser_module
from legal_rag.config import load_settings
from legal_rag.factory import build_orchestrator
from legal_rag.schemas import StandaloneQuestion, Turn

HERE = Path(__file__).parent
REPEATS = 2
HAND_DATE = datetime(2026, 1, 1)


class Recording:
    def __init__(self, model):
        self.model = model
        self.raw: list[str | None] = []
        self.prompts: list[str] = []

    def __getattr__(self, name):
        return getattr(self.model, name)

    def generate(self, messages, **kwargs):
        response = self.model.generate(messages, **kwargs)
        if kwargs.get("response_schema") is StandaloneQuestion:
            parsed = response.parsed
            self.raw.append(parsed.question if isinstance(parsed, StandaloneQuestion) else None)
            self.prompts.append(messages[-1].content)
        return response


def recorded_inputs(path: Path) -> list[dict]:
    items = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        for index, turn in enumerate(row["turns"]):
            if turn.get("condensed_raw") is None or not turn.get("condense_input"):
                continue  # the model was not called (no history): nothing to compare
            items.append({"input_id": f"{row['id']}#{index + 1}", "source": "recorded", "kind": row["kind"],
                          "message": turn["condense_input"]["message"],
                          "history": [Turn.model_validate(t) for t in turn["condense_input"]["history"]]})
    return items


def hand_inputs() -> list[dict]:
    items = []
    for case in json.loads((HERE / "referent_cases.json").read_text()):
        history = [Turn(role="user", content=case["user"], created_at=HAND_DATE),
                   Turn(role="assistant", content=case["assistant"], created_at=HAND_DATE, evidence_ids=["hand-written"])]
        items.append({"input_id": case["id"], "source": "hand", "kind": f"referent_{case['where']}",
                      "message": case["message"], "history": history, "referent_any": case["referent_any"]})
    return items


def main() -> None:
    record = Path(sys.argv[1])
    tag = time.strftime("%Y%m%dT%H%M%S")
    orch = build_orchestrator(load_settings())
    condenser = orch.condenser
    recorder = Recording(condenser.model)
    condenser.model = recorder
    fix_view = condenser_module.condenser_view
    arms = {"base": lambda turns: list(turns), "fix": fix_view}
    items = recorded_inputs(record) + hand_inputs()
    out_dir = HERE / "runs"
    out_dir.mkdir(exist_ok=True)
    manifest = {"tag": tag, "record": str(record), "legal_rag": str(Path(legal_rag.__file__).parent),
                "model": getattr(recorder.model, "name", None), "history_turns": condenser.history_turns,
                "inputs": len(items), "repeats": REPEATS, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out_dir / f"{tag}-replay.manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    with (out_dir / f"{tag}-replay.jsonl").open("w") as fh:
        for repeat in range(REPEATS):
            for item in items:
                for arm, view in arms.items():
                    condenser_module.condenser_view = view
                    recorder.raw.clear()
                    recorder.prompts.clear()
                    t0 = time.perf_counter()
                    final = condenser.condense(item["message"], item["history"])
                    ms = round((time.perf_counter() - t0) * 1000)
                    raw = recorder.raw[0] if recorder.raw else None
                    row = {"input_id": item["input_id"], "source": item["source"], "kind": item["kind"], "arm": arm,
                           "repeat": repeat, "message": item["message"], "raw": raw, "final": final,
                           "replaced": raw is not None and raw.strip() != final, "ms": ms,
                           "prompt": recorder.prompts[0] if recorder.prompts else None,
                           "referent_any": item.get("referent_any"),
                           # the stored turns of the window in full, for the blind labellers (packet.py)
                           "window": [f"- {'사용자' if t.role == 'user' else '상담'}: {t.content}"
                                      for t in item["history"][-condenser.history_turns:]]}
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                    print(repeat, item["input_id"], arm, ms, (raw or "")[:60], flush=True)
    condenser_module.condenser_view = fix_view
    print("tag", tag)


if __name__ == "__main__":
    main()

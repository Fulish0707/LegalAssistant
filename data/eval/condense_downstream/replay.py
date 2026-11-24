"""Downstream condenser check, turn replay (PROTOCOL.md, main judgment).

For every condenser input of the recorded base run (the message and the stored history the
condenser was given), runs the whole turn again under three arms on an in-memory copy of that
history, so all arms answer the same message after the same conversation:
  base    switch off (serving behaviour)
  base2   switch off again (noise of the answer model, temperature 0.2)
  message rag.generate_from_message on (the answer model gets the user's message as the question)
The arm order rotates per input. Nothing is written to the conversation store on disk.

Usage (serving .env loaded, cwd = repo root, <tree> = worktree at the fix commit):
  ( set -a; . ./.env; set +a; exec env -u PYTHONHOME PYTHONPATH=<tree>/src .venv/bin/python \\
      data/eval/condense_downstream/replay.py <recorded base run .jsonl> )
Writes runs/<tag>-replay.jsonl.
"""
import json
import sys
import time
from pathlib import Path

import legal_rag
from legal_rag.config import load_settings
from legal_rag.factory import build_orchestrator
from legal_rag.rag.conversation import InMemoryConversationStore
from legal_rag.schemas import Turn

HERE = Path(__file__).parent
ARMS = {
    "base": {"generate_from_message": False},
    "base2": {"generate_from_message": False},
    "message": {"generate_from_message": True},
}


def recorded_inputs(path: Path) -> list[dict]:
    items = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        for index, turn in enumerate(row["turns"]):
            if turn.get("condensed_raw") is None or not turn.get("condense_input"):
                continue  # no history, or the model was not called: the condenser plays no part
            items.append({"input_id": f"{row['id']}#{index + 1}", "kind": row["kind"],
                          "message": turn["condense_input"]["message"],
                          "history": [Turn.model_validate(t) for t in turn["condense_input"]["history"]],
                          "condensed_raw": turn["condensed_raw"]})
    return items


def main() -> None:
    record = Path(sys.argv[1])
    tag = time.strftime("%Y%m%dT%H%M%S")
    settings = load_settings()
    orch = build_orchestrator(settings)
    items = recorded_inputs(record)
    out_dir = HERE / "runs"
    out_dir.mkdir(exist_ok=True)
    manifest = {"tag": tag, "record": str(record), "legal_rag": str(Path(legal_rag.__file__).parent),
                "model": getattr(orch.generator.model, "name", None), "history_turns": settings.rag.history_turns,
                "inputs": len(items), "arms": ARMS, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out_dir / f"{tag}-replay.manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    names = list(ARMS)
    with (out_dir / f"{tag}-replay.jsonl").open("w") as fh:
        for number, item in enumerate(items):
            order = names[number % len(names):] + names[:number % len(names)]
            for arm in order:
                orch.settings = settings.model_copy(update={"rag": settings.rag.model_copy(update=ARMS[arm])})
                orch.store = InMemoryConversationStore()
                for turn in item["history"]:
                    orch.store.append("replay", turn)
                t0 = time.perf_counter()
                response = orch.handle(item["message"], conversation_id="replay")
                seconds = round(time.perf_counter() - t0, 2)
                trace = response.trace
                row = {
                    "input_id": item["input_id"], "kind": item["kind"], "arm": arm, "message": item["message"],
                    "condensed_raw": item["condensed_raw"],
                    "window": [f"- {'사용자' if t.role == 'user' else '상담'}: {t.content}"
                               for t in item["history"][-settings.rag.history_turns:]],
                    "response_kind": response.kind, "reason": response.reason, "seconds": seconds,
                    "search_queries": trace.search_queries if trace else [],
                    "timings": {t.stage: round(t.ms) for t in trace.timings} if trace else {},
                    "regenerations": trace.regenerations if trace else None,
                    "evidence_ids": trace.evidence_ids if trace else [],
                    "verification_ok": response.verification.ok if response.verification else None,
                    "answer": response.answer.model_dump(mode="json") if response.answer else None,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                print(item["input_id"], arm, response.kind, seconds, flush=True)
    orch.settings = settings
    print("tag", tag)


if __name__ == "__main__":
    main()

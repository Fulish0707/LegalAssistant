"""Condenser history check, end-to-end layer (PROTOCOL.md): run scenarios through the serving
orchestrator, record the condenser's raw output next to the query actually used, the exact input the
condenser received (message and stored history), and apply the fixed checks K1-K4.
A copy of data/eval/condense_facts/run.py that also records condenser inputs for replay.py.

Usage (serving .env loaded, cwd = repo root so data paths resolve):
  ( set -a; . ./.env; set +a; exec env -u PYTHONHOME PYTHONPATH=<tree>/src .venv/bin/python \
      data/eval/condense_summary/run.py <arm> <scenarios.json> [ids...] )

<tree> is a worktree at the code under test (base: the preregistration commit). Conversation ids are
mt-<tag>-<id>; remove them afterwards with cleanup.py.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import legal_rag
from legal_rag.config import load_settings
from legal_rag.factory import build_orchestrator
from legal_rag.schemas import StandaloneQuestion

HERE = Path(__file__).parent
# The diagnostic's fixed checks K1-K4, loaded by path (both scripts are named run.py).
_spec = importlib.util.spec_from_file_location("multiturn_check_run", HERE.parent / "multiturn_check" / "run.py")
_diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_diag)
answer_text, check, evidence_text = _diag.answer_text, _diag.check, _diag.evidence_text


class Recording:
    """Proxy for the condenser's model: keeps what the model wrote before any code-side check."""

    def __init__(self, model):
        self.model = model
        self.raw: list[str | None] = []

    def __getattr__(self, name):
        return getattr(self.model, name)

    def generate(self, messages, **kwargs):
        response = self.model.generate(messages, **kwargs)
        if kwargs.get("response_schema") is StandaloneQuestion:
            parsed = response.parsed
            self.raw.append(parsed.question if isinstance(parsed, StandaloneQuestion) else None)
        return response


class InputLog:
    """Wraps the condenser's condense(): keeps the message and the full stored history it was given."""

    def __init__(self, condense):
        self.condense = condense
        self.calls: list[dict] = []

    def __call__(self, message, history):
        self.calls.append({"message": message, "history": [turn.model_dump(mode="json") for turn in history]})
        return self.condense(message, history)


def main() -> None:
    arm, scenario_path, *only = sys.argv[1:]
    scenarios = json.loads(Path(scenario_path).read_text())
    tag = time.strftime("%Y%m%dT%H%M%S")
    settings = load_settings()
    orch = build_orchestrator(settings)
    recorder = Recording(orch.condenser.model)
    orch.condenser.model = recorder
    inputs = InputLog(orch.condenser.condense)
    orch.condenser.condense = inputs
    out_dir = HERE / "runs"
    out_dir.mkdir(exist_ok=True)
    stem = f"{tag}-{arm}-{Path(scenario_path).stem}"
    manifest = {"tag": tag, "arm": arm, "scenarios": scenario_path, "legal_rag": str(Path(legal_rag.__file__).parent),
                "model": getattr(orch.generator.model, "name", None), "history_turns": settings.rag.history_turns,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out_dir / f"{stem}.manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    with (out_dir / f"{stem}.jsonl").open("w") as fh:
        for scenario in scenarios:
            if only and scenario["id"] not in only:
                continue
            conversation_id = f"mt-{tag}-{scenario['id']}"
            turns = []
            last = None
            for message in scenario["turns"]:
                recorder.raw.clear()
                inputs.calls.clear()
                t0 = time.perf_counter()
                last = orch.handle(message, conversation_id=conversation_id)
                trace = last.trace
                condensed = trace.search_queries[0] if trace and trace.search_queries else None
                raw = recorder.raw[0] if recorder.raw else None
                turns.append({
                    "message": message,
                    "kind": last.kind,
                    "seconds": round(time.perf_counter() - t0, 2),
                    "questions": last.questions,
                    "condensed_raw": raw,
                    "condensed": condensed,
                    "replaced": bool(recorder.raw) and (raw or "").strip() != (condensed or ""),
                    "search_queries": trace.search_queries if trace else [],
                    "timings": {t.stage: round(t.ms) for t in trace.timings} if trace else {},
                    "evidence_headers": [item.header for item in last.evidence.items] if last.evidence else [],
                    "evidence_text_chars": len(evidence_text(last)),
                    "answer": last.answer.model_dump(mode="json") if last.answer else None,
                    "answer_text": answer_text(last),
                    "reason": last.reason,
                    "condense_input": inputs.calls[0] if inputs.calls else None,
                })
            row = {"id": scenario["id"], "kind": scenario["kind"], "arm": arm, "conversation_id": conversation_id,
                   "checks": check(scenario, last), "turns": turns}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(scenario["id"], row["checks"], [t["kind"] for t in turns], [t["replaced"] for t in turns], flush=True)
    print("tag", tag)


if __name__ == "__main__":
    main()

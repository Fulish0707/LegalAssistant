"""Multi-turn path diagnostic (PROTOCOL.md): run each scenario's turns through the serving
orchestrator in one conversation and apply the fixed checks K1-K4 to the last turn.

Run with the serving .env loaded:
  ( set -a; . ./.env; set +a; exec env -u PYTHONPATH -u PYTHONHOME .venv/bin/python data/eval/multiturn_check/run.py )

Conversation ids are mt-<run tag>-<scenario>; they land in the serving conversation store and
are removed by cleanup.py after a backup.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

from legal_rag.config import load_settings
from legal_rag.factory import build_orchestrator

HERE = Path(__file__).parent
SCENARIOS = json.loads((HERE / "scenarios.json").read_text())


def answer_text(response) -> str:
    a = response.answer
    if a is None:
        return ""
    return "\n".join([a.summary, a.analysis, *a.possible_actions])


def evidence_text(response) -> str:
    if response.evidence is None:
        return ""
    return "\n".join(json.dumps(item.header, ensure_ascii=False) + "\n" + item.text for item in response.evidence.items)


def check(scenario: dict, last) -> dict:
    condensed = last.trace.search_queries[0] if last.trace and last.trace.search_queries else ""
    k1 = any(word in condensed for word in scenario["condensed_any"])
    if scenario.get("condensed_none"):
        k1 = k1 and not any(word in condensed for word in scenario["condensed_none"])
    evidence = evidence_text(last)
    answer = answer_text(last)
    return {
        "K1": k1,
        "K2": any(word in evidence for word in scenario["evidence_any"]),
        "K3": any(word in answer for word in scenario["answer_any"]),
        "K4": last.kind == "answer",
    }


def main() -> None:
    only = set(sys.argv[1:])
    tag = time.strftime("%Y%m%dT%H%M%S")
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    settings = load_settings()
    orch = build_orchestrator(settings)
    out_dir = HERE / "runs"
    out_dir.mkdir(exist_ok=True)
    manifest = {"tag": tag, "git": sha, "model": getattr(orch.generator.model, "name", None),
                "store": str(getattr(orch.store, "path", None)), "history_turns": settings.rag.history_turns,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out_dir / f"{tag}.manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    with (out_dir / f"{tag}.jsonl").open("w") as fh:
        for scenario in SCENARIOS:
            if only and scenario["id"] not in only:
                continue
            conversation_id = f"mt-{tag}-{scenario['id']}"
            turns = []
            last = None
            for message in scenario["turns"]:
                t0 = time.perf_counter()
                last = orch.handle(message, conversation_id=conversation_id)
                trace = last.trace
                turns.append({
                    "message": message,
                    "kind": last.kind,
                    "seconds": round(time.perf_counter() - t0, 2),
                    "questions": last.questions,
                    "condensed": trace.search_queries[0] if trace and trace.search_queries else None,
                    "search_queries": trace.search_queries if trace else [],
                    "needs_more_info": trace.analysis.needs_more_info if trace and trace.analysis else None,
                    "missing_facts": trace.analysis.missing_facts if trace and trace.analysis else [],
                    "timings": {t.stage: round(t.ms) for t in trace.timings} if trace else {},
                    "evidence_headers": [item.header for item in last.evidence.items] if last.evidence else [],
                    "answer": last.answer.model_dump(mode="json") if last.answer else None,
                    "reason": last.reason,
                })
            row = {"id": scenario["id"], "kind": scenario["kind"], "conversation_id": conversation_id,
                   "checks": check(scenario, last), "turns": turns}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(scenario["id"], row["checks"], [t["kind"] for t in turns], flush=True)


if __name__ == "__main__":
    main()

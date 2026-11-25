"""Run the quantity relation detector on stored answers.

  python detect.py dev               # fictional held-out sets (development: read freely)
  python detect.py eval --counts     # eval set: counts only (rule not frozen yet)
  python detect.py eval              # eval set: per-sentence output (only after _key/rule.FROZEN exists)
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from legal_rag.verification.quantities import check_quantities, extract_quantities  # noqa: E402

HERE = Path(__file__).parent
DEV_SETS = ["answer_generation", "answer_prompt_heldout", "answer_prompt_heldout2",
            "adapter_consult_heldout", "adapter_consult_heldout2", "adapter_consult_heldout3"]
EVAL_DIR = ROOT / "data/eval/reports/chat_pipeline/numeric-eval-2026-11-23"
FINAL_TEST = ROOT / "data/eval/reports/chat_pipeline/final-test-2026-10-06"


def dev_rows():
    for name in DEV_SETS:
        d = ROOT / "data/eval" / name
        cases = {c["id"]: c for c in json.loads((d / "cases.json").read_text())}
        sources = {s["id"]: s for s in json.loads((d / "sources.json").read_text())}
        for results in sorted((d / "runs").rglob("results.jsonl")):
            for line in results.read_text().splitlines():
                r = json.loads(line)
                if r.get("status") != "completed" or not r.get("body"):
                    continue
                case = cases[r["case_id"]]
                evidence = [sources[s]["quote"] for s in case["sources"]]
                yield {"id": f"{name}/{results.parent.name}/{r['case_id']}", "answer": r["body"],
                       "evidence": evidence, "user": case["message"]}


def eval_rows():
    questions = {}
    for s in ("formal", "everyday"):
        for line in (FINAL_TEST / f"{s}.questions.jsonl").read_text().splitlines():
            q = json.loads(line)
            questions[f"{q['dataset']}:{q['id']}"] = q["query"]
    for s in ("formal", "everyday"):
        for line in (EVAL_DIR / s / "verify.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["kind"] != "answer" or not r.get("final_text"):
                continue
            yield {"id": r["key"], "answer": r["final_text"], "evidence": r.get("evidence", []), "user": questions[r["key"]]}


def main() -> None:
    mode = sys.argv[1]
    counts_only = "--counts" in sys.argv
    if mode == "eval" and not counts_only and not (HERE / "_key/rule.FROZEN").exists():
        raise SystemExit("eval sentences are hidden until the rule is frozen (_key/rule.FROZEN)")
    rows = list(dev_rows() if mode == "dev" else eval_rows())
    answers = flagged = sentences = 0
    by_rule = Counter()
    for row in rows:
        answers += 1
        sentences += len({q.start for q in extract_quantities(row["answer"])})
        findings = check_quantities(row["answer"], row["evidence"], row["user"])
        flagged += bool(findings)
        for f in findings:
            by_rule[f.rule] += 1
            if not counts_only:
                print(f"[{f.rule}] {row['id']} | {f.quantity} | {f.detail}\n    {f.sentence[:220]}")
    print(f"\nanswers {answers} | quantity mentions {sentences} | flagged answers {flagged} "
          f"({flagged / max(answers, 1):.1%}) | findings {dict(by_rule)}")


if __name__ == "__main__":
    main()

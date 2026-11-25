"""Development scan (read-only): which facts in recorded condenser outputs are not in the conversation?

Reads every condensed turn of the multi-turn runs recorded so far and prints, per output, the
quantities (dates, periods, amounts, counts, rates) absent from the user's messages, the statute
articles and case numbers absent from the whole window, and law-name-like words absent from the
window. The window is rebuilt the way the orchestrator stores it: the user's messages, and for an
assistant turn its summary + analysis (answer) or its questions (clarification).

Usage: python dev_scan.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from legal_rag.textutil.refs import extract_case_refs, extract_statute_refs  # noqa: E402
from legal_rag.verification.quantities import extract_quantities  # noqa: E402

RUNS = [ROOT / "data/eval/multiturn_check/runs/20261124T091824.jsonl",
        *sorted((ROOT / "data/eval/multiturn_fix/runs").glob("*.jsonl"))]
LAW_WORD = re.compile(r"[가-힣·]{2,}(?:법률|특례법|기본법|보호법|시행령|시행규칙|법)(?=[\s,.?에의상을이가과와은는도에서]|$)")
NOT_LAWS = {"방법", "불법", "위법", "적법", "합법", "편법", "입법", "사법", "법"}


def assistant_text(turn: dict) -> str:
    if turn["answer"]:
        return f"{turn['answer']['summary']}\n{turn['answer']['analysis']}"
    return "\n".join(turn.get("questions") or [])


def scan(output: str, users: list[str], window: str) -> dict:
    user_text = "\n".join(users)
    user_keys = {q.key for q in extract_quantities(user_text)}
    quantities = [q.value for q in extract_quantities(output) if q.key not in user_keys]
    articles = {r.article for r in extract_statute_refs(window)}
    statutes = [r.render() for r in extract_statute_refs(output) if r.article not in articles]
    cases = [c for c in extract_case_refs(output) if c not in set(extract_case_refs(window))]
    squashed = "".join(window.split())
    laws = [w for w in LAW_WORD.findall(output) if w not in NOT_LAWS and "".join(w.split()) not in squashed]
    return {"quantities": quantities, "statutes": statutes, "cases": cases, "law_words": laws}


def main() -> None:
    total = flagged = 0
    for path in RUNS:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            users: list[str] = []
            window: list[str] = []
            for turn in row["turns"]:
                raw = turn.get("condensed_raw", turn.get("condensed")) if users else None
                if raw:
                    total += 1
                    found_cur = scan(raw, [*users, turn["message"]], "\n".join([*window, turn["message"]]))
                    if any(found_cur.values()):
                        flagged += 1
                        print(f"{path.stem[:30]:30} {row['id']} t{len(users)} {found_cur}")
                        print("    ", raw)
                users.append(turn["message"])
                window += [turn["message"], assistant_text(turn)]
    print(f"condensed outputs: {total}, with an unsupported fact: {flagged}")


if __name__ == "__main__":
    main()

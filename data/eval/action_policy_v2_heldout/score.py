"""Score the preregistered metrics of README.md for every run under runs/.

python score.py  -> one table per code version (old-*, new-*) and model
"""
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
CASES = {c["id"]: c for c in json.loads((HERE / "cases.json").read_text())}
NOT_ANSWER = {"V3-no-agreement-statutes-only", "V4-not-stated", "V6-statement-conflicts-messages",
              "V7-agreed-claim-only", "V9-clause-rule-missing"}
ANSWER = {"V1-agreed-rule-and-finding", "V2-agreed-found-on-research", "V5-agreement-only-in-document",
          "V8-clause-rule-available"}


def candidate_rows(run):
    for line in (run / "results.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["arm"] == "candidate":
            yield row


def outcome(row):
    response = row.get("response") or {}
    decisions = (response.get("trace") or {}).get("post_retrieval", [])
    return response.get("kind", row["status"]), [d["action"] for d in decisions], [d.get("reason") for d in decisions]


def matches(case, kind, actions):
    if CASES[case].get("policy_enabled") is False:
        return kind == "answer" and actions == []
    return actions == CASES[case]["manual_expected_actions"]


def main(out=sys.stdout):
    groups = collections.defaultdict(list)
    for run in sorted((HERE / "runs").iterdir()):
        code, rest = run.name.split("-", 2)[0] + "-" + run.name.split("-", 2)[1], run.name.split("-", 2)[2]
        model = rest.rsplit("-rep", 1)[0]
        groups[(code, model)].append(run)
    for (code, model), runs in sorted(groups.items()):
        m1 = m2 = m4 = decisions = invalid = n = 0
        kinds = collections.defaultdict(list)
        detail = []
        for run in runs:
            for row in candidate_rows(run):
                case = row["case_id"]
                kind, actions, reasons = outcome(row)
                n += 1
                ok = matches(case, kind, actions)
                m2 += ok
                m1 += case in NOT_ANSWER and kind == "answer"
                m4 += case in ANSWER and kind != "answer"
                decisions += len(reasons)
                invalid += reasons.count("assessment_failed")
                kinds[case].append(kind)
                detail.append((case, run.name.rsplit("-rep", 1)[1], kind, "→".join(actions) or "-",
                               ",".join(r for r in reasons if r) or "-", "✓" if ok else "✗"))
        stable = sum(len(set(k)) == 1 for k in kinds.values())
        print(f"\n## {code} / {model} ({len(runs)} run(s), {n} candidate arms)", file=out)
        print(f"- M1 unsafe answers: {m1} / {len(NOT_ANSWER) * len(runs)}", file=out)
        print(f"- M2 action sequence match: {m2} / {n}", file=out)
        print(f"- M3 invalid assessments: {invalid} / {decisions} decisions", file=out)
        print(f"- M4 missed answers: {m4} / {len(ANSWER) * len(runs)}", file=out)
        print(f"- same final kind in every run: {stable} / {len(kinds)} cases", file=out)
        print("\n| case | rep | kind | actions | reasons | match |\n|---|---|---|---|---|---|", file=out)
        for case, rep, kind, actions, reasons, ok in sorted(detail):
            print(f"| {case} | {rep} | {kind} | {actions} | {reasons} | {ok} |", file=out)


if __name__ == "__main__":
    main()

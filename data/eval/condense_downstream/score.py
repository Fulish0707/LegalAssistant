"""Gates of PROTOCOL.md from the turn replay and the blind labels. Read-only.

Usage: python score.py <replay .jsonl> <key .json> <condense labels A> <condense labels B> \\
                       <pair labels A> <pair labels B>
Condense labels map C-ids to "Q" | "QA" | "A" (two labels that differ take the worse).
Pair labels map P-ids to "1" | "2" | "=" ; a fix wins (loses) only when both labellers prefer it
(the base); anything else is a tie.
"""
import json
import math
import statistics
import sys
from pathlib import Path

RANK = {"Q": 0, "QA": 1, "A": 2}
FIXES = {"message": 0.5}  # allowed rise of the median turn time, seconds


def main() -> None:
    replay_path, key_path, cond_a, cond_b, pair_a, pair_b = sys.argv[1:]
    rows = [json.loads(line) for line in Path(replay_path).read_text().splitlines()]
    key = json.loads(Path(key_path).read_text())
    ca, cb, pa, pb = (json.loads(Path(p).read_text()) for p in (cond_a, cond_b, pair_a, pair_b))

    condense = {key["condense"][i]: max(ca[i], cb[i], key=lambda mark: RANK[mark]) for i in key["condense"]}
    condense_disagreements = sorted(i for i in key["condense"] if ca[i] != cb[i])
    defect = sorted(i for i, mark in condense.items() if mark != "Q")
    good = sorted(i for i, mark in condense.items() if mark == "Q")

    outcome: dict[tuple[str, str], str] = {}
    pair_disagreements = []
    for item, entry in key["pairs"].items():
        fix_side = "2" if entry["base_side"] == 1 else "1"
        base_side = str(entry["base_side"])
        votes = (pa[item], pb[item])
        if votes[0] != votes[1]:
            pair_disagreements.append(item)
        if votes == (fix_side, fix_side):
            result = "win"
        elif votes == (base_side, base_side):
            result = "loss"
        else:
            result = "tie"
        outcome[(entry["input_id"], entry["arm"])] = result

    def tally(arm, inputs):
        results = [outcome[(i, arm)] for i in inputs]
        return {"win": results.count("win"), "loss": results.count("loss"), "tie": results.count("tie")}

    def margin(t):
        return t["win"] - t["loss"]

    by_arm = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    no_answer = {arm: sum(row["response_kind"] != "answer" for row in arm_rows) for arm, arm_rows in by_arm.items()}
    failed_check = {arm: sum(row["verification_ok"] is False for row in arm_rows) for arm, arm_rows in by_arm.items()}
    median_s = {arm: statistics.median(row["seconds"] for row in arm_rows) for arm, arm_rows in by_arm.items()}

    noise_d, noise_all = tally("base2", defect), tally("base2", defect + good)
    report = {"inputs": len(condense), "defect_inputs": defect, "condense_labels": condense,
              "condense_disagreements": condense_disagreements, "pair_disagreements": len(pair_disagreements),
              "noise_base2": {"defect": noise_d, "all": noise_all}, "no_answer": no_answer,
              "verification_failed": failed_check, "median_turn_seconds": median_s, "fixes": {}}
    for fix, slack in FIXES.items():
        d, g = tally(fix, defect), tally(fix, good)
        threshold = max(3, math.ceil(len(defect) / 3))
        gates = {
            "G0 defects reproduced (|D| >= 6)": len(defect) >= 6,
            "G0 noise on D (|base2 margin| <= 2)": abs(margin(noise_d)) <= 2,
            f"G1 margin on D >= {threshold}": margin(d) >= threshold,
            "G1 losses on D <= 1": d["loss"] <= 1,
            "G2 losses - wins on good inputs <= 2": -margin(g) <= 2,
            "G3 no-answer turns <= base + 1": no_answer[fix] <= no_answer["base"] + 1,
            "G3 failed verifications <= base + 1": failed_check[fix] <= failed_check["base"] + 1,
            f"G4 median turn time <= base + {slack} s": median_s[fix] <= median_s["base"] + slack,
        }
        report["fixes"][fix] = {"defect": d, "good": g,
                                "per_input": {i: outcome[(i, fix)] for i in defect + good},
                                "gates": gates, "adopt_before_G5": all(gates.values())}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Gates G1-G6 of PROTOCOL.md from the runs and the two blind label files. Read-only.

Usage: python score.py <replay .jsonl> <key .json> <labels A .json> <labels B .json> \\
                       <e2e base .jsonl> <e2e fix .jsonl> <e2e base .jsonl> <e2e fix .jsonl>
A label file maps item id -> "Q" | "QA" | "A". Two labels that differ take the worse (A > QA > Q).
"""
import json
import re
import statistics
import sys
from pathlib import Path

SOURCE_HANDLE = re.compile(r"\[(?:LEGAL SOURCE\s+)?S\d+(?:\s*,\s*S\d+)*\]")
RANK = {"Q": 0, "QA": 1, "A": 2}


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main() -> None:
    replay_path, key_path, label_a, label_b, *e2e = sys.argv[1:]
    replay = load_jsonl(replay_path)
    key = json.loads(Path(key_path).read_text())
    first, second = (json.loads(Path(p).read_text()) for p in (label_a, label_b))
    disagreements = sorted(i for i in key if first[i] != second[i])
    merged = {i: max(first[i], second[i], key=lambda mark: RANK[mark]) for i in key}
    by_text = {(entry["input_id"], entry["text"]): merged[item] for item, entry in key.items()}

    # Per input and arm: the worse label over the repeats.
    label: dict[tuple[str, str], str] = {}
    for row in replay:
        if row["raw"] is None:
            continue
        mark = by_text[(row["input_id"], row["raw"].strip())]
        cell = (row["input_id"], row["arm"])
        label[cell] = max(label.get(cell, "Q"), mark, key=lambda mark: RANK[mark])
    inputs = sorted({row["input_id"] for row in replay})
    count = {arm: {mark: sum(label.get((i, arm)) == mark for i in inputs) for mark in RANK} for arm in ("base", "fix")}
    base_defects = count["base"]["A"] + count["base"]["QA"]
    fix_defects = count["fix"]["A"] + count["fix"]["QA"]
    q_to_a = [i for i in inputs if label.get((i, "base")) == "Q" and label.get((i, "fix")) == "A"]
    q_to_qa = [i for i in inputs if label.get((i, "base")) == "Q" and label.get((i, "fix")) == "QA"]
    replaced = {arm: sum(row["replaced"] for row in replay if row["arm"] == arm) for arm in ("base", "fix")}

    summary_cases = sorted({row["input_id"] for row in replay if row["kind"] == "referent_summary"})
    analysis_cases = sorted({row["input_id"] for row in replay if row["kind"] == "referent_analysis"})

    def keeps(case, arm):
        rows = [row for row in replay if row["input_id"] == case and row["arm"] == arm]
        return all(any(word in (row["raw"] or "") for word in row["referent_any"]) for row in rows)

    replay_handles = sum(bool(SOURCE_HANDLE.search(row["final"] or "")) for row in replay if row["arm"] == "fix")

    runs = [load_jsonl(path) for path in e2e]
    arm_of = [run[0]["arm"] for run in runs]
    k = {arm: {name: sum(row["checks"][name] for run, a in zip(runs, arm_of) if a == arm for row in run)
               for name in ("K1", "K2", "K3", "K4")} for arm in ("base", "fix")}
    condense_ms = {arm: [t["timings"].get("condense") for run, a in zip(runs, arm_of) if a == arm for row in run
                         for t in row["turns"] if t["condensed_raw"] is not None and t["timings"].get("condense") is not None]
                   for arm in ("base", "fix")}
    e2e_handles = sum(bool(SOURCE_HANDLE.search(t["condensed"] or "")) for run, a in zip(runs, arm_of) if a == "fix"
                      for row in run for t in row["turns"])
    medians = {arm: statistics.median(values) for arm, values in condense_ms.items()}

    gates = {
        "G1 reproduced (base A+QA >= 4)": base_defects >= 4,
        "G1 fix A <= 1": count["fix"]["A"] <= 1,
        "G1 fix A+QA <= base/2": fix_defects <= base_defects // 2,
        "G2 base Q -> fix A == 0": not q_to_a,
        "G2 base Q -> fix QA <= 1": len(q_to_qa) <= 1,
        "G2 fix replacements <= base": replaced["fix"] <= replaced["base"],
        "G3 summary referents kept 4/4": all(keeps(case, "fix") for case in summary_cases) and len(summary_cases) == 4,
        "G4 no handles in fix final queries": replay_handles == 0 and e2e_handles == 0,
        "G5 K1-K4 fix >= base - 2": all(k["fix"][n] >= k["base"][n] - 2 for n in k["base"]),
        "G6 condense median fix <= base": medians["fix"] <= medians["base"],
    }
    report = {
        "inputs": len(inputs), "labels": count, "label_disagreements": disagreements,
        "base_Q_to_fix_A": q_to_a, "base_Q_to_fix_QA": q_to_qa, "replacements": replaced,
        "summary_referents": {c: {"base": keeps(c, "base"), "fix": keeps(c, "fix")} for c in summary_cases},
        "analysis_referents": {c: {"base": keeps(c, "base"), "fix": keeps(c, "fix")} for c in analysis_cases},
        "handles": {"replay_fix_final": replay_handles, "e2e_fix": e2e_handles},
        "e2e_arms": arm_of, "K": k, "condense_median_ms": medians,
        "per_input": {i: {"base": label.get((i, "base")), "fix": label.get((i, "fix"))} for i in inputs},
        "gates": gates, "adopt": all(gates.values()),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

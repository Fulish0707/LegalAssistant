"""Before/after aggregation for the answer-prompt change (G1 vs G1v2, mixed blind batch).

Usage: python3 compare_prompt_versions.py [scores.jsonl]
Defaults to prompt_compare/scores.jsonl. Reports overall rates, per-scorer paired
deltas (scorer severity cancels within a batch), per-case labels, and an
all-unclear-flipped sensitivity check. Development measurement, not adoption evidence.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
ITEMS = ["C1", "C2", "C3", "C4", "C5"]


def rate(labels):
    applicable = [x for x in labels if x != "해당 없음"]
    return sum(x == "맞음" for x in applicable) / len(applicable) if applicable else None


def rates(entries, key):
    per = defaultdict(lambda: defaultdict(list))
    for e in entries:
        version = key[e["packet_id"]]["version"]
        for item in ITEMS:
            per[version][item].append(e[item])
        per[version]["C6"].append(e["C6"])
    out = {}
    for version, items in per.items():
        out[version] = {item: rate(items[item]) for item in ITEMS}
        out[version]["C6"] = sum(items["C6"]) / len(items["C6"])
    return out


def main():
    pc = HERE / "prompt_compare"
    key = {json.loads(l)["packet_id"]: json.loads(l) for l in (pc / "_key.jsonl").read_text().splitlines()}
    scores_path = Path(sys.argv[1]) if len(sys.argv) > 1 else pc / "scores.jsonl"
    scores = [json.loads(l) for l in scores_path.read_text().splitlines()]
    assert len(scores) == 48 and {e["packet_id"] for e in scores} == set(key)
    overall = rates(scores, key)
    print("## 전체 (개발용 측정)\n")
    print("| 버전 | C1 | C2 | C3 | C4 | C5 | C6 |")
    print("|---|---|---|---|---|---|---|")
    for version in ("g1", "g1v2"):
        row = overall[version]
        print(f"| {version} | " + " | ".join(f"{row[i]:.2f}" for i in ITEMS) + f" | {row['C6']:.2f} |")
    print("\n## 채점자별 짝 비교 (g1v2 - g1, 같은 채점자 안)\n")
    print("| 채점자 | C1 | C2 | C3 | C4 | C5 | C6 |")
    print("|---|---|---|---|---|---|---|")
    batches = defaultdict(list)
    for e in scores:
        batches[e.get("batch", "?")].append(e)
    deltas = []
    for batch in sorted(batches):
        r = rates(batches[batch], key)
        if "g1" not in r or "g1v2" not in r:
            continue
        delta = {i: r["g1v2"][i] - r["g1"][i] for i in ITEMS}
        delta["C6"] = r["g1v2"]["C6"] - r["g1"]["C6"]
        deltas.append(delta)
        print(f"| {batch} | " + " | ".join(f"{delta[i]:+.2f}" for i in ITEMS) + f" | {delta['C6']:+.2f} |")
    print("\n## 사례별 (g1 -> g1v2, 반복 2회)\n")
    per_case = defaultdict(lambda: defaultdict(list))
    for e in scores:
        meta = key[e["packet_id"]]
        per_case[meta["case_id"]][meta["version"]].append("/".join(str(e[i]) for i in ITEMS) + f"/{e['C6']}")
    print("| 사례 | g1 | g1v2 |")
    print("|---|---|---|")
    for cid in sorted(per_case):
        print(f"| {cid} | {' ; '.join(per_case[cid]['g1'])} | {' ; '.join(per_case[cid]['g1v2'])} |")
    flipped = []
    for e in scores:
        f = dict(e)
        if e.get("unclear"):
            for i in ITEMS:
                f[i] = {"맞음": "틀림", "틀림": "맞음"}.get(f[i], f[i])
        flipped.append(f)
    fr = rates(flipped, key)
    print("\n## 애매 항목 전부 뒤집기 민감도 (g1v2 - g1)\n")
    for i in ITEMS + ["C6"]:
        base = overall["g1v2"][i] - overall["g1"][i]
        flip = fr["g1v2"][i] - fr["g1"][i]
        changed = " 방향 바뀜" if (base < 0) != (flip < 0) else ""
        print(f"- {i}: {base:+.2f} -> {flip:+.2f}{changed}")
    summary = {"overall": overall, "per_scorer_deltas": deltas,
               "unclear": sum(bool(e.get("unclear")) for e in scores)}
    (pc / "compare_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

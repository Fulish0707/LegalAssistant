"""Aggregate the answer-generation comparison: script metrics + blind scores.

Usage: python3 score.py [scores.jsonl]   (default: scores.jsonl next to this file)
Prints markdown tables and writes score_summary.json. Descriptive statistics only.
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
ARMS = ["g1-exaone", "g2-base", "g3-legal", "g3s-legal"]
LABEL_ITEMS = ["C1", "C2", "C3", "C4", "C5"]


def load_runs():
    rows = []
    for run_dir in sorted((HERE / "runs").iterdir()):
        results = run_dir / "results.jsonl"
        if not results.exists():
            continue
        tag = run_dir.name.rsplit("-rep", 1)[0]
        for line in results.read_text().splitlines():
            row = json.loads(line)
            row["arm_tag"] = tag
            rows.append(row)
    return rows


def script_metrics(rows):
    print("## 자동 지표\n")
    print("| 갈래 | 시도 | 완료 | 실패 | A1 검증 통과 | A2 치명 답변 | A3 지연 중앙값(ms) | 본문 길이 중앙값 |")
    print("|---|---|---|---|---|---|---|---|")
    summary = {}
    for arm in ARMS:
        group = [r for r in rows if r["arm_tag"] == arm]
        done = [r for r in group if r["status"] == "completed"]
        ok = sum(r["verification"]["ok"] for r in done)
        fatal = [r for r in done if r.get("fatal")]
        elapsed = statistics.median(r["elapsed_ms"] for r in done) if done else 0
        length = statistics.median(len(r["body"]) for r in done) if done else 0
        summary[arm] = {"attempted": len(group), "completed": len(done), "failed": len(group) - len(done),
                        "verification_ok": ok, "fatal_answers": len(fatal),
                        "fatal_cases": sorted({r["case_id"] for r in fatal}),
                        "median_elapsed_ms": round(elapsed), "median_body_chars": round(length)}
        print(f"| {arm} | {len(group)} | {len(done)} | {len(group) - len(done)} | {ok}/{len(done)} | "
              f"{len(fatal)} | {elapsed:.0f} | {length:.0f} |")
    return summary


def blind_scores(rows, scores_path):
    key = {json.loads(line)["packet_id"]: json.loads(line) for line in (HERE / "_key.jsonl").read_text().splitlines()}
    scores = {}
    for line in Path(scores_path).read_text().splitlines():
        entry = json.loads(line)
        scores[entry["packet_id"]] = entry
    missing = set(key) - set(scores)
    if missing:
        print(f"WARNING: {len(missing)} packets unscored")
    per_arm = defaultdict(lambda: defaultdict(list))
    per_case = defaultdict(list)
    unclear = []
    for pid, entry in scores.items():
        meta = key[pid]
        arm = meta["arm"]
        for item in LABEL_ITEMS:
            per_arm[arm][item].append(entry[item])
        per_arm[arm]["C6"].append(entry["C6"])
        per_case[(meta["case_id"], arm)].append(entry)
        if entry.get("unclear"):
            unclear.append((pid, meta, entry))
    print("\n## 블라인드 채점 (맞음 비율, 해당 없음 제외)\n")
    print("| 갈래 | C1 | C2 | C3 | C4 | C5 | C6 평균 | 애매 |")
    print("|---|---|---|---|---|---|---|---|")
    rates = {}
    for arm in ARMS:
        cells = []
        rates[arm] = {}
        for item in LABEL_ITEMS:
            labels = per_arm[arm][item]
            applicable = [x for x in labels if x != "해당 없음"]
            rate = sum(x == "맞음" for x in applicable) / len(applicable) if applicable else None
            rates[arm][item] = rate
            na = len(labels) - len(applicable)
            cells.append(f"{rate:.2f} ({sum(x == '맞음' for x in applicable)}/{len(applicable)})" + (f" [제외 {na}]" if na else "") if rate is not None else "해당 없음")
        c6 = per_arm[arm]["C6"]
        rates[arm]["C6"] = statistics.mean(c6) if c6 else None
        n_unclear = sum(1 for _, meta, _ in unclear if meta["arm"] == arm)
        print(f"| {arm} | " + " | ".join(cells) + f" | {rates[arm]['C6']:.2f} | {n_unclear} |")
    print("\n## 사례별 (반복 2회 라벨)\n")
    print("| 사례 | " + " | ".join(ARMS) + " |")
    print("|---|---|---|---|---|")
    case_ids = sorted({meta["case_id"] for meta in key.values()})
    for cid in case_ids:
        cells = []
        for arm in ARMS:
            entries = per_case.get((cid, arm), [])
            cells.append("; ".join("/".join(str(e[i]) for i in LABEL_ITEMS) + f"/{e['C6']}" for e in entries) or "-")
        print(f"| {cid} | " + " | ".join(cells) + " |")
    return rates, per_arm, unclear, key, scores


def main():
    rows = load_runs()
    summary = script_metrics(rows)
    scores_path = sys.argv[1] if len(sys.argv) > 1 else HERE / "scores.jsonl"
    if Path(scores_path).exists():
        rates, per_arm, unclear, key, scores = blind_scores(rows, scores_path)
        summary["blind_rates"] = rates
        summary["unclear_packets"] = [{"packet_id": pid, "arm": meta["arm"], "case_id": meta["case_id"]}
                                      for pid, meta, _ in unclear]
        print("\n## 주 비교\n")
        for a, b, name in (("g3-legal", "g1-exaone", "G3 어댑터 대 G1 현재"), ("g3-legal", "g2-base", "G3 어댑터 대 G2 원본")):
            for item in ("C1", "C2", "C3", "C4"):
                ra, rb = rates[a].get(item), rates[b].get(item)
                if ra is not None and rb is not None:
                    print(f"- {name} {item}: {ra:.2f} 대 {rb:.2f}")
        fatal_rule = summary["g3-legal"]["fatal_answers"] > summary["g1-exaone"]["fatal_answers"]
        print(f"- 채택 차단 규칙(A2 치명이 G1보다 많음): {'발동' if fatal_rule else '미발동'}")
    (HERE / "score_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Held-out check for the answer prompt change: packets, then gate evaluation.

  python3 heldout_check.py packets          # mixed blind packets (old+new), 4 batches of 8
  python3 heldout_check.py gates scores.jsonl   # metrics + preregistered gates (README.md)
"""
import json
import random
import statistics
import sys
import uuid
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
SEED = 20261123
ITEMS = ["C1", "C2", "C3", "C4", "C5"]


def runs():
    out = []
    for run_dir in sorted((HERE / "runs").iterdir()):
        if not (run_dir / "results.jsonl").exists():
            continue
        version = "new" if run_dir.name.startswith("new-") else "old"
        for line in (run_dir / "results.jsonl").read_text().splitlines():
            row = json.loads(line)
            row["version"], row["run"] = version, run_dir.name
            out.append(row)
    return out


def evidence_text(case, by_id):
    blocks = []
    for sid in case["sources"]:
        s = by_id[sid]
        if s["kind"] == "statute":
            blocks.append(f"[법령] {s['law_name']} {s['article']} (시행 {s['effective_date']})\n{s['quote']}")
        else:
            blocks.append(f"[가상 판결문 판단 창] 사건 {s['case_number']} {s['title']}\n{s['quote']}")
    return "\n\n".join(blocks)


def build_packets():
    rows = [r for r in runs() if r["status"] == "completed"]
    assert len(rows) == 32, len(rows)
    packets, key = [], []
    for row in rows:
        pid = uuid.uuid4().hex[:12]
        packets.append({"packet_id": pid, "case_id": row["case_id"], "body": row["body"]})
        key.append({"packet_id": pid, "case_id": row["case_id"], "version": row["version"], "run": row["run"]})
    random.Random(SEED).shuffle(packets)
    out = HERE / "scoring"
    out.mkdir(exist_ok=True)
    (out / "_key.jsonl").write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in key))
    for i in range(4):
        chunk = packets[i * 8:(i + 1) * 8]
        (out / f"packets-{i + 1}.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in chunk))
    cases = json.loads((HERE / "cases.json").read_text())
    by_id = {s["id"]: s for s in json.loads((HERE / "sources.json").read_text())}
    comparison = json.loads((HERE / "comparison.json").read_text())
    materials = [{"case_id": c["id"], "message": c["message"], "evidence": evidence_text(c, by_id),
                  "comparison": comparison[c["id"]]} for c in cases]
    (out / "scoring_materials.json").write_text(json.dumps(materials, ensure_ascii=False, indent=2) + "\n")
    print("32 packets -> scoring/packets-1..4.jsonl (8 each); key kept separate")


def counts(entries, key):
    per = defaultdict(lambda: defaultdict(list))
    for e in entries:
        v = key[e["packet_id"]]["version"]
        for i in ITEMS:
            per[v][i].append(e[i])
        per[v]["C6"].append(e["C6"])
    out = {}
    for v, d in per.items():
        out[v] = {}
        for i in ITEMS:
            app = [x for x in d[i] if x != "해당 없음"]
            out[v][i] = (sum(x == "맞음" for x in app), len(app))
        out[v]["C6"] = statistics.mean(d["C6"])
    return out


def gates(scores_path):
    key = {json.loads(line)["packet_id"]: json.loads(line)
           for line in (HERE / "scoring/_key.jsonl").read_text().splitlines()}
    scores = [json.loads(line) for line in Path(scores_path).read_text().splitlines()]
    assert len(scores) == 32 and {e["packet_id"] for e in scores} == set(key)
    rows = runs()
    print("## 자동 지표\n")
    print("| 버전 | 완료 | 인용 검증 통과 | 근거 밖 인용 | 지연 중앙값 | 본문 길이 중앙값 |")
    print("|---|---|---|---|---|---|")
    auto = {}
    for v in ("old", "new"):
        g = [r for r in rows if r["version"] == v]
        done = [r for r in g if r["status"] == "completed"]
        auto[v] = {"completed": len(done), "attempted": len(g),
                   "verify_fail": sum(not r["verification"]["ok"] for r in done),
                   "fatal": sum(bool(r.get("fatal")) for r in done)}
        print(f"| {v} | {len(done)}/{len(g)} | {len(done) - auto[v]['verify_fail']}/{len(done)} | {auto[v]['fatal']} | "
              f"{statistics.median(r['elapsed_ms'] for r in done) / 1000:.1f}초 | {statistics.median(len(r['body']) for r in done):.0f}자 |")
    c = counts(scores, key)
    print("\n## 블라인드 채점\n")
    print("| 버전 | C1 | C2 | C3 | C4 | C5 | C6 |")
    print("|---|---|---|---|---|---|---|")
    for v in ("old", "new"):
        print(f"| {v} | " + " | ".join(f"{c[v][i][0]}/{c[v][i][1]}" for i in ITEMS) + f" | {c[v]['C6']:.2f} |")
    batches = defaultdict(list)
    for e in scores:
        batches[e["batch"]].append(e)
    print("\n## 채점자별 짝 비교 (new − old 맞음 비율)\n")
    print("| 채점자 | C1 | C2 | C3 | C4 | C5 |")
    print("|---|---|---|---|---|---|")
    c3_nonneg = 0
    for b in sorted(batches):
        bc = counts(batches[b], key)
        if "old" not in bc or "new" not in bc:
            print(f"| {b} | (한쪽 버전 없음) |")
            continue
        d = {}
        for i in ITEMS:
            (no, nn), (oo, on) = bc["new"][i], bc["old"][i]
            d[i] = (no / nn if nn else 0) - (oo / on if on else 0)
        c3_nonneg += d["C3"] >= 0
        print(f"| {b} | " + " | ".join(f"{d[i]:+.2f}" for i in ITEMS) + " |")
    rate = lambda v, i: c[v][i][0] / c[v][i][1] if c[v][i][1] else 0  # noqa: E731
    g1 = rate("new", "C3") > rate("old", "C3") and rate("new", "C4") >= rate("old", "C4")
    g2 = c3_nonneg >= 3
    g3 = all(c["new"][i][0] >= c["old"][i][0] - 2 for i in ("C1", "C2", "C5"))
    g4 = auto["new"]["fatal"] <= auto["old"]["fatal"] and auto["new"]["verify_fail"] <= auto["old"]["verify_fail"]
    print("\n## 채택 게이트\n")
    print(f"- G1 주효과(C3 new>old, C4 new>=old): {'통과' if g1 else '실패'} "
          f"(C3 {rate('old', 'C3'):.2f}→{rate('new', 'C3'):.2f}, C4 {rate('old', 'C4'):.2f}→{rate('new', 'C4'):.2f})")
    print(f"- G2 일관성(C3 변화>=0 채점자 3명 이상): {'통과' if g2 else '실패'} ({c3_nonneg}/4)")
    print(f"- G3 회귀(C1·C2·C5 new >= old-2): {'통과' if g3 else '실패'} "
          + ", ".join(f"{i} {c['old'][i][0]}→{c['new'][i][0]}" for i in ("C1", "C2", "C5")))
    print(f"- G4 안전: {'통과' if g4 else '실패'}")
    print(f"- **판정: {'채택' if all((g1, g2, g3, g4)) else '채택하지 않음'}**")
    flipped = []
    for e in scores:
        f = dict(e)
        if e.get("unclear"):
            for i in ITEMS:
                f[i] = {"맞음": "틀림", "틀림": "맞음"}.get(f[i], f[i])
        flipped.append(f)
    fc = counts(flipped, key)
    fr = lambda v, i: fc[v][i][0] / fc[v][i][1] if fc[v][i][1] else 0  # noqa: E731
    print(f"\n## 애매 {sum(bool(e.get('unclear')) for e in scores)}건 전부 뒤집기 민감도\n")
    for i in ITEMS:
        print(f"- {i}: {rate('new', i) - rate('old', i):+.2f} -> {fr('new', i) - fr('old', i):+.2f}")
    (HERE / "gate_summary.json").write_text(json.dumps(
        {"auto": auto, "counts": c, "gates": {"G1": g1, "G2": g2, "G3": g3, "G4": g4},
         "adopted": all((g1, g2, g3, g4))}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "packets":
        build_packets()
    else:
        gates(sys.argv[2])

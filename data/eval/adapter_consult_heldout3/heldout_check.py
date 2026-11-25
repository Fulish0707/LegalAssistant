"""Held-out check for the consulting adapter: blind packets, then preregistered gates (README.md).

  python3 heldout_check.py packets              # 48 mixed blind packets, 4 batches of 12
  python3 heldout_check.py gates scores.jsonl   # metrics + gates
Run directories are named <arm>-<label>-repN with arm in E, A0, A4.
"""
import json
import random
import statistics
import sys
import uuid
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
SEED = 20261126
ARMS = ("E", "A0", "A4")
ITEMS = ["C1", "C2", "C3", "C4", "C5"]
PER_BATCH = 12


def runs():
    out = []
    for run_dir in sorted((HERE / "runs").iterdir()):
        if not (run_dir / "results.jsonl").exists():
            continue
        arm = run_dir.name.split("-")[0]
        assert arm in ARMS, run_dir.name
        for line in (run_dir / "results.jsonl").read_text().splitlines():
            row = json.loads(line)
            row["arm"], row["run"] = arm, run_dir.name
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
    packets, key = [], []
    for row in rows:
        pid = uuid.uuid4().hex[:12]
        packets.append({"packet_id": pid, "case_id": row["case_id"], "body": row["body"]})
        key.append({"packet_id": pid, "case_id": row["case_id"], "arm": row["arm"], "run": row["run"]})
    random.Random(SEED).shuffle(packets)
    out = HERE / "scoring"
    out.mkdir(exist_ok=True)
    (out / "_key.jsonl").write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in key))
    batches = (len(packets) + PER_BATCH - 1) // PER_BATCH
    for i in range(batches):
        chunk = packets[i * PER_BATCH:(i + 1) * PER_BATCH]
        (out / f"packets-{i + 1}.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in chunk))
    cases = json.loads((HERE / "cases.json").read_text())
    by_id = {s["id"]: s for s in json.loads((HERE / "sources.json").read_text())}
    comparison = json.loads((HERE / "comparison.json").read_text())
    materials = [{"case_id": c["id"], "message": c["message"], "evidence": evidence_text(c, by_id),
                  "comparison": comparison[c["id"]]} for c in cases]
    (out / "scoring_materials.json").write_text(json.dumps(materials, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(packets)} packets -> scoring/packets-1..{batches}.jsonl; key kept separate")


def counts(entries, key):
    per = defaultdict(lambda: defaultdict(list))
    for e in entries:
        a = key[e["packet_id"]]["arm"]
        for i in ITEMS:
            per[a][i].append(e[i])
        per[a]["C6"].append(e["C6"])
    out = {}
    for a, d in per.items():
        out[a] = {i: (sum(x == "맞음" for x in d[i]), sum(x != "해당 없음" for x in d[i])) for i in ITEMS}
        out[a]["C6"] = statistics.mean(d["C6"])
    return out


def gates(scores_path):
    key = {json.loads(line)["packet_id"]: json.loads(line)
           for line in (HERE / "scoring/_key.jsonl").read_text().splitlines()}
    scores = [json.loads(line) for line in Path(scores_path).read_text().splitlines()]
    assert {e["packet_id"] for e in scores} == set(key) and len(scores) == len(key)
    rows = runs()
    print("## 자동 지표\n\n| 갈래 | 완료 | 인용 검증 통과 | 근거 밖 인용 | 지연 중앙값 | 본문 길이 중앙값 |\n|---|---|---|---|---|---|")
    auto = {}
    for a in ARMS:
        g = [r for r in rows if r["arm"] == a]
        done = [r for r in g if r["status"] == "completed"]
        auto[a] = {"completed": len(done), "attempted": len(g),
                   "verify_fail": sum(not r["verification"]["ok"] for r in done),
                   "fatal": sum(bool(r.get("fatal")) for r in done)}
        med = lambda xs: statistics.median(xs) if xs else 0  # noqa: E731
        print(f"| {a} | {len(done)}/{len(g)} | {len(done) - auto[a]['verify_fail']}/{len(done)} | {auto[a]['fatal']} | "
              f"{med([r['elapsed_ms'] for r in done]) / 1000:.1f}초 | {med([len(r['body']) for r in done]):.0f}자 |")
    c = counts(scores, key)
    print("\n## 블라인드 채점\n\n| 갈래 | C1 | C2 | C3 | C4 | C5 | C6 |\n|---|---|---|---|---|---|---|")
    for a in ARMS:
        print(f"| {a} | " + " | ".join(f"{c[a][i][0]}/{c[a][i][1]}" for i in ITEMS) + f" | {c[a]['C6']:.2f} |")
    rate = lambda a, i: c[a][i][0] / c[a][i][1] if c[a][i][1] else 0  # noqa: E731
    h1 = rate("A4", "C3") > rate("A0", "C3") and rate("A4", "C4") >= rate("A0", "C4")
    h2 = all(c["A4"][i][0] >= c["A0"][i][0] - 2 for i in ("C1", "C2", "C5"))
    h3 = (auto["A4"]["fatal"] <= auto["A0"]["fatal"] and auto["A4"]["verify_fail"] <= auto["A0"]["verify_fail"]
          and auto["A4"]["completed"] >= auto["A0"]["completed"])
    decision1 = h1 and h2 and h3
    q1 = all(c["A4"][i][0] >= c["E"][i][0] - 2 for i in ITEMS)
    q2 = auto["A4"]["fatal"] <= auto["E"]["fatal"] and auto["A4"]["verify_fail"] <= auto["E"]["verify_fail"]
    decision2 = decision1 and q1 and q2
    ok = lambda b: "통과" if b else "실패"  # noqa: E731
    print("\n## 판정 1 — A4가 A0를 대체하는가\n")
    print(f"- H1 주효과: {ok(h1)} (C3 {rate('A0', 'C3'):.2f}→{rate('A4', 'C3'):.2f}, C4 {rate('A0', 'C4'):.2f}→{rate('A4', 'C4'):.2f})")
    print(f"- H2 회귀: {ok(h2)} " + ", ".join(f"{i} {c['A0'][i][0]}→{c['A4'][i][0]}" for i in ("C1", "C2", "C5")))
    print(f"- H3 안전: {ok(h3)}")
    print(f"- **판정 1: {'대체' if decision1 else '대체하지 않음'}**")
    print("\n## 판정 2 — A4가 EXAONE급인가\n")
    print(f"- Q1 (C1~C5 A4 ≥ E−2): {ok(q1)} " + ", ".join(f"{i} E {c['E'][i][0]} / A4 {c['A4'][i][0]}" for i in ITEMS))
    print(f"- Q2 안전: {ok(q2)}")
    print(f"- **판정 2: {'EXAONE급(서빙 교체 후보, 패리티·공존 측정 남음)' if decision2 else 'EXAONE 유지'}**")
    batches = defaultdict(list)
    for e in scores:
        batches[e["batch"]].append(e)
    print("\n## 채점자별 A4 − A0 맞음 비율\n\n| 채점자 | " + " | ".join(ITEMS) + " |\n|---|---|---|---|---|---|")
    for b in sorted(batches):
        bc = counts(batches[b], key)
        if "A0" not in bc or "A4" not in bc:
            print(f"| {b} | (한쪽 갈래 없음) |")
            continue
        cells = []
        for i in ITEMS:
            (x, n), (y, m) = bc["A4"][i], bc["A0"][i]
            cells.append(f"{(x / n if n else 0) - (y / m if m else 0):+.2f}")
        print(f"| {b} | " + " | ".join(cells) + " |")
    flipped = []
    for e in scores:
        f = dict(e)
        if e.get("unclear"):
            for i in ITEMS:
                f[i] = {"맞음": "틀림", "틀림": "맞음"}.get(f[i], f[i])
        flipped.append(f)
    fc = counts(flipped, key)
    fr = lambda a, i: fc[a][i][0] / fc[a][i][1] if fc[a][i][1] else 0  # noqa: E731
    print(f"\n## 애매 {sum(bool(e.get('unclear')) for e in scores)}건 전부 뒤집기 민감도 (A4 − A0, A4 − E)\n")
    for i in ITEMS:
        print(f"- {i}: {rate('A4', i) - rate('A0', i):+.2f} -> {fr('A4', i) - fr('A0', i):+.2f}; "
              f"{rate('A4', i) - rate('E', i):+.2f} -> {fr('A4', i) - fr('E', i):+.2f}")
    (HERE / "gate_summary.json").write_text(json.dumps(
        {"auto": auto, "counts": c, "gates": {"H1": h1, "H2": h2, "H3": h3, "Q1": q1, "Q2": q2},
         "decision1_replace_A0": decision1, "decision2_exaone_level": decision2}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "packets":
        build_packets()
    else:
        gates(sys.argv[2])

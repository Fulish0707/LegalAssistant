"""Build blind scoring packets from the answer-generation runs.

Packets carry only what a scorer may see: the answer body (no citations/confidence),
the user question, and the evidence the answer was written from. Arm/model/rep go to
_key.jsonl, never into packets. Fixed seed shuffle.

Run: python3 data/eval/answer_generation/build_packets.py
"""
import json
import random
import uuid
from pathlib import Path

HERE = Path(__file__).parent
SEED = 20261120


def evidence_text(case, by_id):
    blocks = []
    for sid in case["sources"]:
        source = by_id[sid]
        if source["kind"] == "statute":
            blocks.append(f"[법령] {source['law_name']} {source['article']} (시행 {source['effective_date']})\n{source['quote']}")
        else:
            blocks.append(f"[가상 판결문 판단 창] 사건 {source['case_number']} {source['title']}\n{source['quote']}")
    return "\n\n".join(blocks)


def main():
    cases = {c["id"]: c for c in json.loads((HERE / "cases.json").read_text())}
    by_id = {s["id"]: s for s in json.loads((HERE / "sources.json").read_text())}
    comparison = json.loads((HERE / "comparison.json").read_text())
    packets, key = [], []
    for run_dir in sorted((HERE / "runs").iterdir()):
        results = run_dir / "results.jsonl"
        if not results.exists():
            continue
        tag = run_dir.name.rsplit("-rep", 1)[0]
        rep = run_dir.name.rsplit("-rep", 1)[1]
        for line in results.read_text().splitlines():
            row = json.loads(line)
            if row["status"] != "completed":
                continue
            pid = uuid.uuid4().hex[:12]
            packets.append({"packet_id": pid, "case_id": row["case_id"], "body": row["body"]})
            key.append({"packet_id": pid, "case_id": row["case_id"], "arm": tag,
                        "model": row["model"], "rep": rep, "run": run_dir.name})
    random.Random(SEED).shuffle(packets)
    with (HERE / "packets.jsonl").open("w") as f:
        for packet in packets:
            f.write(json.dumps(packet, ensure_ascii=False) + "\n")
    with (HERE / "_key.jsonl").open("w") as f:
        for entry in key:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    materials = [{"case_id": cid, "message": case["message"], "evidence": evidence_text(case, by_id),
                  "comparison": comparison[cid]} for cid, case in cases.items()]
    (HERE / "scoring_materials.json").write_text(json.dumps(materials, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(packets)} packets, {len(materials)} case materials; key kept separate")


if __name__ == "__main__":
    main()

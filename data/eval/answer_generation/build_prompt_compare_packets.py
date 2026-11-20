"""Mixed blind packets for the prompt before/after comparison (G1 vs G1v2).

Old and new answers are shuffled together so each scorer sees both versions without
knowing which is which; scorer severity then cancels in the within-scorer comparison.

Run after runs/g1v2-exaone-rep* exist:
  python3 data/eval/answer_generation/build_prompt_compare_packets.py
"""
import json
import random
import uuid
from pathlib import Path

HERE = Path(__file__).parent
SEED = 20261121
RUNS = ["g1-exaone-rep1", "g1-exaone-rep2", "g1v2-exaone-rep1", "g1v2-exaone-rep2"]


def main():
    packets, key = [], []
    for run in RUNS:
        for line in (HERE / "runs" / run / "results.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["status"] != "completed":
                continue
            pid = uuid.uuid4().hex[:12]
            packets.append({"packet_id": pid, "case_id": row["case_id"], "body": row["body"]})
            key.append({"packet_id": pid, "case_id": row["case_id"],
                        "version": "g1v2" if run.startswith("g1v2") else "g1",
                        "rep": run[-1], "run": run})
    assert len(packets) == 48, len(packets)
    random.Random(SEED).shuffle(packets)
    out = HERE / "prompt_compare"
    out.mkdir(exist_ok=True)
    with (out / "packets.jsonl").open("w") as f:
        for packet in packets:
            f.write(json.dumps(packet, ensure_ascii=False) + "\n")
    with (out / "_key.jsonl").open("w") as f:
        for entry in key:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    for i in range(4):
        chunk = packets[i * 12:(i + 1) * 12]
        with (out / f"packets-{i + 1}.jsonl").open("w") as f:
            for packet in chunk:
                f.write(json.dumps(packet, ensure_ascii=False) + "\n")
    print(f"{len(packets)} packets -> {out}; 4 scorer batches of 12")


if __name__ == "__main__":
    main()

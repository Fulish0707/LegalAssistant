"""Print the last-turn condensed question, evidence headers and answer summary per scenario (read-only)."""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
ids = set(sys.argv[2:])
for line in path.read_text().splitlines():
    row = json.loads(line)
    if ids and row["id"] not in ids:
        continue
    last = row["turns"][-1]
    print("=" * 80)
    print(row["id"], row["kind"], row["checks"], [t["kind"] for t in row["turns"]], [t["seconds"] for t in row["turns"]])
    for t in row["turns"]:
        print("  turn:", t["message"], "| condensed:", t["condensed"], "| clarify:", t["questions"][:2])
    print("  timings:", last["timings"])
    for h in last["evidence_headers"]:
        print("  ev:", {k: v for k, v in h.items() if k != "Section role"})
    if last["answer"] and ids:
        a = last["answer"]
        print("  summary:", a["summary"])
        print("  actions:", a["possible_actions"])
        print("  analysis:", a["analysis"][:1500])

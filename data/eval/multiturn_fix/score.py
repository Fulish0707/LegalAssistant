"""Summarise runs for PROTOCOL.md: K1-K4 counts per run, and every condensed turn (raw model output,
query used, replaced or not) for the reader's Q/F/A/H classification. Read-only.
Usage: python score.py <run.jsonl> [<run.jsonl> ...]"""
import json
import re
import sys
from pathlib import Path

SOURCE_HANDLE = re.compile(r"\[(?:LEGAL SOURCE\s+)?S\d+(?:\s*,\s*S\d+)*\]")  # as in the fix (b11c616)

for path in sys.argv[1:]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    counts = {k: sum(row["checks"][k] for row in rows) for k in ("K1", "K2", "K3", "K4")}
    turns = [(row["id"], i, t) for row in rows for i, t in enumerate(row["turns"]) if t["condensed_raw"] is not None]
    print("=" * 100)
    print(Path(path).name, f"n={len(rows)}", counts,
          f"condensed turns={len(turns)} replaced={sum(t['replaced'] for _, _, t in turns)}",
          f"handles in final query={sum(bool(SOURCE_HANDLE.search(t['condensed'] or '')) for _, _, t in turns)}")
    for row in rows:
        print(" ", row["id"], row["checks"], [t["kind"] for t in row["turns"]])
    for rid, i, t in turns:
        print(f"--- {rid} turn {i} replaced={t['replaced']} condense_ms={t['timings'].get('condense')}")
        print("    raw:  ", t["condensed_raw"])
        if t["replaced"]:
            print("    used: ", t["condensed"])

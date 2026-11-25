"""Back up the conversation store, then delete only one run's conversations (mt-<tag>-*).
Usage: python cleanup.py <store path> <tag>"""
import shutil
import sqlite3
import sys
import time
from pathlib import Path

store, tag = Path(sys.argv[1]), sys.argv[2]
if not tag.strip():
    sys.exit("empty tag would match every diagnostic conversation")
backup = Path.home() / ".hermes/profiles/kei/cache/scratch" / f"conversations-{time.strftime('%Y%m%dT%H%M%S')}.db"
shutil.copy2(store, backup)
pattern = f"mt-{tag}-%"
with sqlite3.connect(store) as db:
    ids = [row[0] for row in db.execute("SELECT DISTINCT conversation_id FROM turns WHERE conversation_id LIKE ?", (pattern,))]
    db.execute("DELETE FROM turns WHERE conversation_id LIKE ?", (pattern,))
print(f"backup {backup}; deleted {len(ids)} conversations: {ids}")

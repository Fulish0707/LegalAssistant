"""Back up the conversation store, then delete only this diagnostic's conversations (mt-<tag>-M..).
Usage: python cleanup.py <store path> <tag>"""
import shutil
import sqlite3
import sys
import time
from pathlib import Path

store, tag = Path(sys.argv[1]), sys.argv[2]
backup = Path.home() / ".hermes/profiles/kei/cache/scratch" / f"conversations-{time.strftime('%Y%m%dT%H%M%S')}.db"
shutil.copy2(store, backup)
pattern = f"mt-{tag}-M%"
with sqlite3.connect(store) as db:
    ids = [row[0] for row in db.execute("SELECT DISTINCT conversation_id FROM turns WHERE conversation_id LIKE ?", (pattern,))]
    db.execute("DELETE FROM turns WHERE conversation_id LIKE ?", (pattern,))
print(f"backup {backup}; deleted {len(ids)} conversations: {ids}")

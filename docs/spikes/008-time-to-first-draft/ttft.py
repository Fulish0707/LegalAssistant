"""Time-to-first-draft on /v1/chat/stream: when each stage starts, when the first draft text arrives, and the end.
New conversation ids per run (the API keeps conversations in the shared SQLite store; ids are tagged ttft-<run>-<q>)."""
import json
import sys
import time
from pathlib import Path

import httpx

F = Path("/home/layke/Documents/MyLLMAgent/data/eval/reports/chat_pipeline/final-test-2026-10-06")
qs = [json.loads(line)["question"] for name in ("formal", "everyday") for line in (F / f"{name}.frozen.jsonl").read_text().splitlines()]
tag = sys.argv[1]
picks = [int(x) for x in sys.argv[2:]] or [5, 40, 77, 110, 170, 205, 233, 266]
c = httpx.Client(base_url="http://127.0.0.1:8000", timeout=600)
out = Path(__file__).with_name(f"ttft-{tag}.jsonl").open("w")
for i in picks:
    t0 = time.perf_counter()
    ev = {"q": i, "stages": {}, "first_draft": None, "draft_chars": 0}
    with c.stream("POST", "/v1/chat/stream", json={"message": qs[i], "conversation_id": f"ttft-{tag}-{i}-{int(time.time())}"}) as r:
        for line in r.iter_lines():
            if not line:
                continue
            e = json.loads(line)
            now = round(time.perf_counter() - t0, 2)
            if e["type"] == "stage":
                ev["stages"].setdefault(e["stage"], now)
            elif e["type"] == "draft":
                ev["first_draft"] = ev["first_draft"] or now
                ev["draft_chars"] += len(e["text"])
            elif e["type"] == "response":
                ev["end"] = now
                ev["kind"] = e["body"].get("kind")
    out.write(json.dumps(ev, ensure_ascii=False) + "\n")
    out.flush()
    print(json.dumps(ev, ensure_ascii=False), flush=True)

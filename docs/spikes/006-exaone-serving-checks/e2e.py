"""End-to-end smoke of the running API (EXAONE, bge-m3 on CPU): 6 questions via /v1/chat, timing, VRAM, model placement."""
import json
import subprocess
import time
from pathlib import Path

import httpx

F = Path("/home/layke/Documents/MyLLMAgent/data/eval/reports/chat_pipeline/final-test-2026-10-06")
qs = [json.loads(line)["question"] for name in ("formal", "everyday") for line in (F / f"{name}.frozen.jsonl").read_text().splitlines()]
qs = qs[0:1] + qs[60:61] + qs[130:131] + qs[160:161] + qs[200:201] + qs[260:261]
c = httpx.Client(base_url="http://127.0.0.1:8000", timeout=600)
o = httpx.Client(base_url="http://localhost:11434")
out = []
for q in qs:
    t = time.time()
    r = c.post("/v1/chat", json={"message": q, "include_trace": True})
    dt = round(time.time() - t, 1)
    b = r.json()
    vram = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    ps = [(m["name"], round(m["size_vram"] / 2**30, 2), round(m["size"] / 2**30, 2)) for m in o.get("/api/ps").json()["models"]]
    ans = b.get("answer") or {}
    text = ans if isinstance(ans, str) else json.dumps(ans, ensure_ascii=False)
    row = {"q": q[:50], "status": r.status_code, "s": dt, "kind": b.get("kind") or b.get("type"), "citations": len(b.get("citations") or []),
           "stage_ms": (b.get("trace") or {}).get("stage_ms"), "vram_mib": int(vram), "ps": ps, "answer_head": text[:160]}
    out.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
Path(__file__).with_name("e2e.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

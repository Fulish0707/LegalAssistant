"""Spike 006: remaining checks before serving EXAONE 3.5 with bge-m3 on CPU.
  A. near-full context: ~10k-token prompt + num_predict 2048 with the fp16 reranker resident
  B. two concurrent answer requests: does Ollama keep one 12288 slot (no reload, no spill)?
  C. upload-style document embedding on CPU: 20 chunks x 1,000 characters, num_ctx 1024, num_gpu 0
"""
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path("/home/layke/Documents/MyLLMAgent")
OUT = Path(__file__).parent
F = ROOT / "data/eval/reports/chat_pipeline/final-test-2026-10-06"
client = httpx.Client(base_url="http://localhost:11434", timeout=900)
M = "exaone3.5:7.8b"
OPT = {"num_ctx": 12288, "temperature": 0.2}


class Poll:
    def __init__(self):
        self.peak, self.stop = 0, False
        self.t = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop:
            o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
            self.peak = max(self.peak, int(o.split()[0]))
            time.sleep(0.2)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop = True
        self.t.join()


def ps():
    return [(m["name"], round(m["size_vram"] / 2**30, 2), round(m["size"] / 2**30, 2), m.get("context_length"), m.get("expires_at", "")[11:19])
            for m in client.get("/api/ps").json().get("models", [])]


def gen(prompt, n):
    r = client.post("/api/generate", json={"model": M, "prompt": prompt, "stream": False, "keep_alive": "10m", "options": {**OPT, "num_predict": n}})
    r.raise_for_status()
    b = r.json()
    return {"prompt_tokens": b["prompt_eval_count"], "out_tokens": b["eval_count"], "total_s": round(b["total_duration"] / 1e9, 1),
            "load_s": round(b.get("load_duration", 0) / 1e9, 2), "decode_tok_s": round(b["eval_count"] / (b["eval_duration"] / 1e9), 1)}


rows = [json.loads(line) for name in ("formal", "everyday") for line in (F / f"{name}.frozen.jsonl").read_text().splitlines()]
texts = [h["chunk"]["text"] for r in rows for h in r["hits"]]
report = {}

for m in client.get("/api/ps").json().get("models", []):
    client.post("/api/generate", json={"model": m["name"], "keep_alive": 0})
time.sleep(3)
client.post("/api/embed", json={"model": "bge-m3", "input": ["보증금"], "keep_alive": "10m", "options": {"num_ctx": 1024, "num_gpu": 0}}).raise_for_status()

import torch  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402

gen("안녕하세요", 4)
enc = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda", max_length=512, local_files_only=True, model_kwargs={"torch_dtype": torch.float16})
enc.predict([(rows[0]["question"], t) for t in texts[:100]], batch_size=16)
report["resident"] = ps()

# A. near-full context
body = ""
for t in texts:
    if len(body) > 17500:
        break
    body += t + "\n\n"
prompt = ("다음 자료 전체를 조문과 판결별로 빠짐없이 아주 자세히 정리하세요. 각 항목마다 요건, 효과, 예외, 기간, 금액을 모두 쓰고 "
          "끝에 사례 분석을 길게 덧붙이세요.\n\n" + body)
with Poll() as p:
    a = gen(prompt, 2048)
    enc.predict([(rows[1]["question"], t) for t in texts[:100]], batch_size=16)
    a.update({"peak_mib": p.peak, "ps": ps()})
report["A_full_context"] = a
print("A", a, flush=True)

# B. two concurrent requests
short = [f"질문: {rows[i]['question']}\n\n자료:\n" + "\n\n".join(h['chunk']['text'] for h in rows[i]['hits'][:6]) + "\n\n자료에 근거해 답하세요." for i in (3, 140)]
with Poll() as p:
    t = time.time()
    with ThreadPoolExecutor(2) as ex:
        res = list(ex.map(lambda s: gen(s, 512), short))
    report["B_concurrent"] = {"wall_s": round(time.time() - t, 1), "each": res, "peak_mib": p.peak, "ps": ps()}
print("B", report["B_concurrent"], flush=True)
for s in short[:1]:
    t = time.time()
    single = gen(s, 512)
report["B_single_ref"] = single

# C. CPU document embedding, upload style
chunks = [t[:1000] for t in texts if len(t) >= 900][:20]
t = time.time()
r = client.post("/api/embed", json={"model": "bge-m3", "input": chunks, "keep_alive": "10m", "options": {"num_ctx": 1024, "num_gpu": 0}})
r.raise_for_status()
report["C_cpu_doc_embed"] = {"chunks": len(chunks), "s": round(time.time() - t, 2), "ps": ps()}
print("C", report["C_cpu_doc_embed"], flush=True)
(OUT / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

"""Spike 005: does EXAONE 3.5 (num_ctx 12288) fit on the 8 GB card next to the fp16 reranker
when the bge-m3 query embedding runs on CPU? Production config and services are not changed.

Steps (system Ollama on :11434, all models unloaded first):
  1. bge-m3 query embedding on GPU vs CPU (options.num_gpu = 0): latency x5 each
  2. load EXAONE at num_ctx 12288 (short prompt), record ollama ps + VRAM
  3. load the fp16 reranker in this process, score 100 real (query, chunk) pairs at max_length 512
  4. with the reranker resident, run EXAONE on a ~9k-token prompt (num_predict 512)
VRAM is polled with nvidia-smi every 0.2 s; peaks are reported per step.
"""
import json
import subprocess
import threading
import time
from pathlib import Path

import httpx

ROOT = Path("/home/layke/Documents/MyLLMAgent")
OUT = Path(__file__).parent
FROZEN = ROOT / "data/eval/reports/chat_pipeline/final-test-2026-10-06/everyday.frozen.jsonl"
client = httpx.Client(base_url="http://localhost:11434", timeout=600)


class VramPoll:
    def __init__(self):
        self.peak, self._stop = 0, False
        self.t = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self._stop:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout.strip()
            self.peak = max(self.peak, int(out.splitlines()[0]))
            time.sleep(0.2)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self._stop = True
        self.t.join()


def vram_now():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    used, total = map(int, out.split(","))
    return used, total


def ps():
    return [(m["name"], round(m["size_vram"] / 2**30, 2), round(m["size"] / 2**30, 2), m.get("context_length"))
            for m in client.get("/api/ps").json().get("models", [])]


def unload_all():
    for m in client.get("/api/ps").json().get("models", []):
        client.post("/api/generate", json={"model": m["name"], "keep_alive": 0})
    time.sleep(3)


report = {}
rows = [json.loads(line) for line in FROZEN.read_text().splitlines()[:6]]
queries = [r["question"] for r in rows]
chunks = [h["chunk"]["text"] for r in rows for h in r["hits"]][:100]
while len(chunks) < 100:
    chunks = chunks + chunks
chunks = chunks[:100]

unload_all()
report["idle_vram"] = vram_now()

# 1. embedding latency
for label, extra in (("gpu", {}), ("cpu", {"num_gpu": 0})):
    unload_all()
    times = []
    for i, q in enumerate(queries[:5] + queries[:1]):
        t = time.time()
        r = client.post("/api/embed", json={"model": "bge-m3", "input": [q], "keep_alive": "10m",
                                            "options": {"num_ctx": 1024, **extra}})
        r.raise_for_status()
        times.append(round(time.time() - t, 3))
    report[f"embed_{label}"] = {"first_s": times[0], "warm_s": times[1:], "ps": ps(), "vram": vram_now()}
    print(label, report[f"embed_{label}"], flush=True)

# leave bge-m3 loaded on CPU (as in the candidate setup) and load EXAONE
with VramPoll() as poll:
    t = time.time()
    client.post("/api/generate", json={"model": "exaone3.5:7.8b", "prompt": "안녕하세요", "stream": False, "keep_alive": "10m",
                                       "options": {"num_ctx": 12288, "num_predict": 8}}).raise_for_status()
    report["exaone_load"] = {"s": round(time.time() - t, 1), "ps": ps(), "vram": vram_now(), "peak": poll.peak}
print("exaone", report["exaone_load"], flush=True)

# 3. reranker
import torch  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402

with VramPoll() as poll:
    t = time.time()
    enc = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda", max_length=512, local_files_only=True,
                       model_kwargs={"torch_dtype": torch.float16})
    load_s = time.time() - t
    times = []
    for q in queries[:3]:
        t = time.time()
        enc.predict([(q, c) for c in chunks], batch_size=16)
        times.append(round(time.time() - t, 2))
    report["rerank"] = {"load_s": round(load_s, 1), "score_100_s": times, "torch_max_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                        "ps": ps(), "vram": vram_now(), "peak": poll.peak}
print("rerank", report["rerank"], flush=True)

# 4. long EXAONE prompt with the reranker resident
long_text = ""
for c in [h["chunk"]["text"] for r in rows for h in r["hits"]]:
    long_text += c + "\n\n"
    if len(long_text) > 14000:
        break
prompt = f"다음 자료를 읽고 핵심 쟁점을 세 문장으로 요약하세요.\n\n{long_text}"
with VramPoll() as poll:
    t = time.time()
    r = client.post("/api/generate", json={"model": "exaone3.5:7.8b", "prompt": prompt, "stream": False, "keep_alive": "10m",
                                           "options": {"num_ctx": 12288, "num_predict": 512, "temperature": 0.2}})
    r.raise_for_status()
    body = r.json()
    report["exaone_long"] = {"s": round(time.time() - t, 1), "prompt_tokens": body.get("prompt_eval_count"),
                             "out_tokens": body.get("eval_count"),
                             "decode_tok_s": round(body["eval_count"] / (body["eval_duration"] / 1e9), 1),
                             "prefill_tok_s": round(body["prompt_eval_count"] / (body["prompt_eval_duration"] / 1e9), 1),
                             "ps": ps(), "vram": vram_now(), "peak": poll.peak}
    # rerank again after the long generation (both resident)
    t = time.time()
    enc.predict([(queries[0], c) for c in chunks], batch_size=16)
    report["rerank_after"] = {"s": round(time.time() - t, 2), "vram": vram_now(), "peak": poll.peak, "ps": ps()}
print("long", report["exaone_long"], flush=True)
print("rerank_after", report["rerank_after"], flush=True)
(OUT / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
unload_all()

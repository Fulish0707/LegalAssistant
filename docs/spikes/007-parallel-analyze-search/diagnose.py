"""Diagnose: does a bge-m3 embed (CPU) wait for a running EXAONE generation in Ollama? And reranker on GPU during decode?"""
import threading
import time

import httpx
import torch
from sentence_transformers import CrossEncoder

c = httpx.Client(base_url="http://localhost:11434", timeout=300)


def embed():
    t = time.perf_counter()
    c.post("/api/embed", json={"model": "bge-m3", "input": ["보증금 반환 청구"], "truncate": False, "keep_alive": "10m",
                               "options": {"num_ctx": 1024, "num_gpu": 0}}).raise_for_status()
    return round(time.perf_counter() - t, 2)


def gen(out):
    t = time.perf_counter()
    c.post("/api/generate", json={"model": "exaone3.5:7.8b", "prompt": "임대차 분쟁 해결 절차를 200자로 설명하세요.", "stream": False,
                                  "keep_alive": "10m", "options": {"num_ctx": 12288, "num_predict": 200, "temperature": 0}}).raise_for_status()
    out["gen_s"] = round(time.perf_counter() - t, 2)


embed()
gen({})
enc = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda", max_length=512, local_files_only=True, model_kwargs={"torch_dtype": torch.float16})
pairs = [("보증금 반환", "임대차 계약이 끝난 뒤 임대인은 보증금을 돌려주어야 한다. " * 20)] * 40
enc.predict(pairs, batch_size=16)
t = time.perf_counter(); enc.predict(pairs, batch_size=16); alone_rr = round(time.perf_counter() - t, 2)
print("alone: embed", embed(), "rerank", alone_rr)
out = {}
th = threading.Thread(target=gen, args=(out,)); th.start(); time.sleep(0.8)
e = embed()
t = time.perf_counter(); enc.predict(pairs, batch_size=16); rr = round(time.perf_counter() - t, 2)
th.join()
print("during generation: embed", e, "rerank", rr, "gen", out["gen_s"])
out2 = {}
gen(out2); print("generation alone", out2["gen_s"])

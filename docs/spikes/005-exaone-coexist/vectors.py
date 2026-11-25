"""Spike 005b: are bge-m3 query vectors from CPU (num_gpu 0) the same as from GPU? Corpus vectors were built on GPU."""
import json
import math
from pathlib import Path

import httpx

client = httpx.Client(base_url="http://localhost:11434", timeout=300)
F = Path("/home/layke/Documents/MyLLMAgent/data/eval/reports/chat_pipeline/final-test-2026-10-06")
qs = [json.loads(line)["question"] for name in ("formal", "everyday") for line in (F / f"{name}.frozen.jsonl").read_text().splitlines()][::5][:50]


def embed(extra):
    for m in client.get("/api/ps").json().get("models", []):
        client.post("/api/generate", json={"model": m["name"], "keep_alive": 0})
    vecs = []
    for q in qs:
        r = client.post("/api/embed", json={"model": "bge-m3", "input": [q], "truncate": False, "options": {"num_ctx": 1024, **extra}})
        r.raise_for_status()
        vecs.append(r.json()["embeddings"][0])
    return vecs


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


gpu, cpu = embed({}), embed({"num_gpu": 0})
sims = sorted(cos(a, b) for a, b in zip(gpu, cpu))
# how different are two *different* queries, for scale
cross = sorted(cos(gpu[i], gpu[i + 1]) for i in range(len(gpu) - 1))
out = {"n": len(sims), "cpu_vs_gpu_min": sims[0], "cpu_vs_gpu_median": sims[len(sims) // 2], "max_abs_diff": max(
    max(abs(x - y) for x, y in zip(a, b)) for a, b in zip(gpu, cpu)), "neighbor_query_cos_max": cross[-1]}
print(json.dumps(out, indent=1))
Path(__file__).with_name("vectors.json").write_text(json.dumps(out, indent=1))

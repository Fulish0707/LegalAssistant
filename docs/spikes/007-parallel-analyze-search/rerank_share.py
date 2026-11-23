"""How much of retrieve_many is the GPU reranker? Times retrieval with and without the rerank step."""
import json
import time
from pathlib import Path

from legal_rag.config import load_settings
from legal_rag.factory import build_orchestrator

F = Path("/home/layke/Documents/MyLLMAgent/data/eval/reports/chat_pipeline/final-test-2026-10-06")
qs = [json.loads(line)["question"] for name in ("formal", "everyday") for line in (F / f"{name}.frozen.jsonl").read_text().splitlines()]
r = build_orchestrator(load_settings()).retriever
reranker = r.reranker
calls = []
orig = reranker.rerank


def timed(*a, **k):
    t = time.perf_counter()
    out = orig(*a, **k)
    calls.append(time.perf_counter() - t)
    return out


reranker.rerank = timed
r.retrieve_many([qs[1]], 5, None, rerank_query=qs[1])
tot, rr = [], []
for i in (5, 40, 77, 110, 170, 205, 233, 266):
    calls.clear()
    t = time.perf_counter()
    r.retrieve_many([qs[i]], 5, None, rerank_query=qs[i])
    tot.append(time.perf_counter() - t)
    rr.append(sum(calls))
print("retrieve s", [round(x, 2) for x in tot])
print("rerank s  ", [round(x, 2) for x in rr])
print("rerank share", round(sum(rr) / sum(tot), 2))

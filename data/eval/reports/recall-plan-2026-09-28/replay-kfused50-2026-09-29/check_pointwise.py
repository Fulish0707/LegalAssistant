#!/usr/bin/env python3
"""Check the replay's one modelling assumption: cross-encoder scores are pointwise.

build_replay.py reranks the k_fused=50 candidates with the scores recorded at k_fused=100. That is exact only if a
chunk's score does not depend on which other chunks share its batch (fp16 padding and batching can shift the last
digits). This script reranks the k_fused=50 and k_fused=100 candidate sets of a fixed sample with the real fp16
cross-encoder, built as serving builds it, and compares scores and top-10 order with the recorded ones and with the
replay. It needs the GPU, so run it while no judge is running (the judge uses ~6.7 of 8 GB).

Run from the worktree root:
    PYTHONPATH=src <python> <this file> [--ip 10] [--mixed 5]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parents[4] / "src"))

import build_replay as br
from legal_rag.evaluation.recall_variants import BASELINE
from legal_rag.reranking.cross_encoder import CrossEncoderReranker
from legal_rag.schemas import ScoredChunk

SEED = 20260929


def candidates(capture: dict, variant) -> list[ScoredChunk]:
    lists = [[ScoredChunk(chunk=h["chunk"], score=h["score"], retriever=h["retriever"]) for h in leg["hits"]]
             for leg in capture["legs"]]
    return variant.fuse(lists, budget=variant.budget)


def main() -> None:
    n_ip = int(sys.argv[sys.argv.index("--ip") + 1]) if "--ip" in sys.argv else 10
    n_mixed = int(sys.argv[sys.argv.index("--mixed") + 1]) if "--mixed" in sys.argv else 5
    datasets = {}
    for row in br.read_jsonl(br.P1 / "target.jsonl"):
        datasets.setdefault(row["dataset"], []).append(row["case_id"])
    rng = random.Random(SEED)
    wanted = set(rng.sample(datasets["ip-200-fanout"], n_ip) + rng.sample(datasets["crossdomain-60-fanout"], n_mixed))

    # Same construction as legal_rag.factory: batch size 16 and max length 512 are the class defaults.
    reranker = CrossEncoderReranker(model="BAAI/bge-reranker-v2-m3", device="cuda", dtype="float16", local_files_only=True)
    rows = []
    for record in br.stream_stages():
        case_id = record["case"]["case_id"]
        if case_id not in wanted:
            continue
        capture = record["capture"]
        recorded = br.RecordedReranker(capture)
        row = {"case_id": case_id, "dataset": record["case"]["dataset"]}
        for label, variant in (("k50", br.K50), ("k100", BASELINE)):
            cands = candidates(capture, variant)
            live = reranker.rerank(capture["query"], cands, 100)
            replayed = recorded.rerank(capture["query"], cands, 100)
            diffs = [abs(h.score - recorded.scores[h.chunk.id]) for h in live]
            live_ids, replay_ids = [h.chunk.id for h in live], [h.chunk.id for h in replayed]
            row[label] = {"candidates": len(cands), "max_abs_score_diff": max(diffs), "mean_abs_score_diff": sum(diffs) / len(diffs),
                          "top10_identical": live_ids[:10] == replay_ids[:10],
                          "top10_shared": len(set(live_ids[:10]) & set(replay_ids[:10])),
                          "full_order_identical": live_ids == replay_ids}
        rows.append(row)
        print(f"{case_id[:14]} {row['dataset'][:5]}  k50: max|d|={row['k50']['max_abs_score_diff']:.2e} top10_identical={row['k50']['top10_identical']}"
              f"  k100: max|d|={row['k100']['max_abs_score_diff']:.2e} top10_identical={row['k100']['top10_identical']}", flush=True)

    summary = {label: {"cases": len(rows), "max_abs_score_diff": max(r[label]["max_abs_score_diff"] for r in rows),
                       "cases_top10_identical": sum(r[label]["top10_identical"] for r in rows),
                       "cases_full_order_identical": sum(r[label]["full_order_identical"] for r in rows),
                       "min_top10_shared": min(r[label]["top10_shared"] for r in rows)} for label in ("k50", "k100")}
    print(json.dumps(summary, indent=1))
    with (BASE / "pointwise-check.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"sample_seed": SEED, "reranker": {"model": "BAAI/bge-reranker-v2-m3", "device": "cuda",
                                 "dtype": "float16", "max_length": 512, "batch_size": 16}, "summary": summary, "cases": rows},
                                indent=2) + "\n")
    print("wrote pointwise-check.json")


if __name__ == "__main__":
    main()

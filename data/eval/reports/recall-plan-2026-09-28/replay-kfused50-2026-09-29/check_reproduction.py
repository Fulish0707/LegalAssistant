#!/usr/bin/env python3
"""Reproduce the 2026-09-19 fan-out top 10 from today's captured legs and recorded fp16 scores.

The 2026-09-19 run used k_fused=50, no leg floor and an fp32 reranker, against an index built earlier. Replaying floor 0
at k_fused=50 from the Priority 1 capture (build_replay.RecordedReranker) uses today's legs and fp16 scores. If the two
top-10 lists agree, fp16 and index drift did not change the ranking, the replay method is sound, and the 09-19 to
current change comes from the floor and k_fused alone. Needs no GPU; judges nothing.

Run from the worktree root:
    PYTHONPATH=src <python> <this file>
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parents[4] / "src"))

import build_replay as br
from legal_rag.evaluation.recall_stages import replay_stages
from legal_rag.evaluation.recall_variants import BASELINE

FLOOR0_K50 = replace(BASELINE, name="floor0_k50", leg_floor=0, budget=50)


def main() -> None:
    previous = {ds: {r["case_id"]: [h["chunk_id"] for h in r["ranked"][:br.POOL_DEPTH]] for r in
                     br.read_jsonl(br.P1 / "grading" / ds / "runs" / f"{br.PREVIOUS}.jsonl")} for ds in br.DATASETS}
    result = {ds: {"cases": 0, "ordered_top10_identical": 0, "top10_set_identical": 0, "mean_shared_chunks": 0.0,
                   "cases_with_a_different_set": []} for ds in br.DATASETS}
    for record in br.stream_stages():
        ds, case_id, capture = record["case"]["dataset"], record["case"]["case_id"], record["capture"]
        hits = replay_stages(capture, br.RecordedReranker(capture), variant=FLOOR0_K50)["reranked"]["hits"]
        replayed, old = [h["chunk_id"] for h in hits[:br.POOL_DEPTH]], previous[ds][case_id]
        out = result[ds]
        out["cases"] += 1
        out["ordered_top10_identical"] += replayed == old
        out["top10_set_identical"] += set(replayed) == set(old)
        out["mean_shared_chunks"] += len(set(replayed) & set(old))
        if set(replayed) != set(old):
            position = {h["chunk_id"]: (rank, h["score"]) for rank, h in enumerate(hits, 1)}
            out["cases_with_a_different_set"].append({
                "case_id": case_id,
                "only_in_replay": [{"chunk_id": c, "rank": position[c][0], "score": position[c][1]} for c in replayed if c not in old],
                "only_in_2026_09_19": [{"chunk_id": c, "replay_rank": position.get(c, (None, None))[0],
                                        "replay_score": position.get(c, (None, None))[1]} for c in old if c not in replayed]})
    for out in result.values():
        out["mean_shared_chunks"] /= out["cases"]
    report = {"variant": {"leg_floor": 0, "budget": 50, "sparse_weight": BASELINE.sparse_weight, "rrf_k": BASELINE.rrf_k},
              "compared_with": "runs/previous_k50.jsonl (2026-09-19 fan-out: k_fused=50, no floor, fp32)", "datasets": result}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    with (BASE / "reproduction-check.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

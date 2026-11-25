#!/usr/bin/env python3
"""Where in the top 10 do the changes land? Precision in ranks 1-5 against ranks 6-10.

Flask builds its evidence from the top-ranked chunks until it holds five distinct documents, so ranks 1-5 are the part
of the top 10 most likely to reach the model. Same pools, judgments, contrasts and bootstrap as analyze_replay.py.

Run from the worktree root, after analyze_replay.py:
    PYTHONPATH=src <python> <this file>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import analyze_replay as ar
from legal_rag.evaluation.graded_metrics import restrict_to_pool
from legal_rag.evaluation.judgment_store import JudgmentStore
from legal_rag.evaluation.relevance_judge import judge_key
from legal_rag.evaluation.relevance_pool import read_jsonl

p1 = ar.p1
BANDS = ("p_top5_grade1", "p_top5_grade2", "p_6to10_grade1", "p_6to10_grade2")


def main() -> None:
    key = judge_key(p1.MODEL, think=True)
    case_meta = p1.load_case_meta()
    out = {}
    for name in ("ip-200-fanout", "crossdomain-60-fanout"):
        pool_dir = ar.BASE / "grading" / name
        pool_rows = read_jsonl(pool_dir / "pool.jsonl")
        store = JudgmentStore(pool_dir / "judgments.jsonl")
        grades, _, failures, _ = restrict_to_pool(store.grades(key), store.documents(key), set(store.failures(key)),
                                                  {(r["case_id"], r["chunk_id"]) for r in pool_rows})
        if failures:
            raise SystemExit(f"{name}: failed parses in the pool")
        runs = {path.stem: read_jsonl(path) for path in sorted((pool_dir / "runs").glob("*.jsonl"))}
        flat = {}
        for mode, rows in runs.items():
            flat[mode] = {}
            for row in rows:
                g = [grades[(row["case_id"], hit["chunk_id"])] for hit in row["ranked"][:10]]   # KeyError if unjudged
                flat[mode][row["case_id"]] = {
                    "p_top5_grade1": sum(x >= 1 for x in g[:5]) / 5, "p_top5_grade2": sum(x == 2 for x in g[:5]) / 5,
                    "p_6to10_grade1": sum(x >= 1 for x in g[5:]) / 5, "p_6to10_grade2": sum(x == 2 for x in g[5:]) / 5}
        out[name] = {
            "means": {mode: {m: p1.avg([r[m] for r in flat[mode].values()]) for m in BANDS} for mode in flat},
            "deltas": {label: {m: ar.contrast(flat, left, right, case_meta, m) for m in BANDS}
                       for label, (left, right) in ar.CONTRASTS.items()}}
        print(f"\n=== {name} ===")
        print(f"{'precision by rank band':26s} {'2026-09-19':>10s} {'replay k50':>10s} {'current':>8s} | {'k_fused 50->100':>26s} | {'floor @k50':>26s}")
        for m in BANDS:
            means, d = out[name]["means"], out[name]["deltas"]
            cell = lambda x: f"{x['delta']:+.3f} [{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}]"
            print(f"{m:26s} {means[ar.PREVIOUS][m]:10.3f} {means[ar.REPLAY][m]:10.3f} {means[ar.CURRENT][m]:8.3f} | "
                  f"{cell(d['k_fused_50_to_100'][m]):>26s} | {cell(d['floor_fp16_drift_at_k50'][m]):>26s}")
    with (BASE / "rank-bands.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print("\nwrote rank-bands.json")


if __name__ == "__main__":
    main()

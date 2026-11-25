#!/usr/bin/env python3
"""How much of the evidence Flask hands the model is relevant, for each run?

analyze_replay.py counts a question as covered when at least one verified grade-2 quote is visible; that saturates near
0.86 and cannot see a denser evidence set. This measures the density directly: pass each run's top 10 through Flask's
formatter (five documents, 1,200 characters each) and count the shown documents that contain a visible chunk graded
>= 1 (or 2). A document counts only if such a chunk kept text inside the 1,200 characters.

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
from legal_rag.evaluation.recall_report import prepare_evidence
from legal_rag.evaluation.relevance_judge import judge_key
from legal_rag.evaluation.relevance_pool import read_jsonl

p1 = ar.p1
METRICS = ("shown_documents", "relevant_documents_grade1", "relevant_documents_grade2",
           "relevant_share_grade1", "relevant_share_grade2")


def per_case(ranked: list[dict], case_id: str, text: dict, grades: dict) -> dict[str, float]:
    evidence = prepare_evidence([{"chunk_id": h["chunk_id"], "document_id": h["document_id"],
                                  "chunk": {"text": text[(case_id, h["chunk_id"])]}} for h in ranked[:10]])
    grade1 = sum(any(grades[(case_id, s["chunk_id"])] >= 1 for s in doc["segments"]) for doc in evidence)
    grade2 = sum(any(grades[(case_id, s["chunk_id"])] == 2 for s in doc["segments"]) for doc in evidence)
    shown = len(evidence)
    return {"shown_documents": shown, "relevant_documents_grade1": grade1, "relevant_documents_grade2": grade2,
            "relevant_share_grade1": grade1 / shown, "relevant_share_grade2": grade2 / shown}


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
        text = {(r["case_id"], r["chunk_id"]): r["text"] for r in pool_rows}
        runs = {path.stem: read_jsonl(path) for path in sorted((pool_dir / "runs").glob("*.jsonl"))}
        flat = {mode: {row["case_id"]: per_case(row["ranked"], row["case_id"], text, grades) for row in rows}
                for mode, rows in runs.items()}
        out[name] = {
            "means": {mode: {m: p1.avg([r[m] for r in flat[mode].values()]) for m in METRICS} for mode in flat},
            "deltas": {label: {m: ar.contrast(flat, left, right, case_meta, m) for m in METRICS}
                       for label, (left, right) in ar.CONTRASTS.items()}}
        print(f"\n=== {name} ===")
        print(f"{'evidence Flask shows':30s} {'2026-09-19':>10s} {'replay k50':>10s} {'current':>8s} | {'k_fused 50->100':>26s} | {'floor @k50':>26s} | {'total':>26s}")
        cell = lambda x: f"{x['delta']:+.3f} [{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}]"
        for m in METRICS:
            means, d = out[name]["means"], out[name]["deltas"]
            print(f"{m:30s} {means[ar.PREVIOUS][m]:10.3f} {means[ar.REPLAY][m]:10.3f} {means[ar.CURRENT][m]:8.3f} | "
                  f"{cell(d['k_fused_50_to_100'][m]):>26s} | {cell(d['floor_fp16_drift_at_k50'][m]):>26s} | {cell(d['total'][m]):>26s}")
    with (BASE / "evidence-precision.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print("\nwrote evidence-precision.json")


if __name__ == "__main__":
    main()

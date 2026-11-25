#!/usr/bin/env python3
"""Paired analysis of the k_fused=50 replay against the two Priority 1 runs.

Three runs share one judged pool per dataset:
  previous_k50        2026-09-19 fan-out: k_fused=50, no floor, fp32 reranker (recorded earlier)
  replay_k50          current settings (floor 3, fp16 scores) re-fused at k_fused=50 (build_replay.py)
  flask_current_k100  current served settings, k_fused=100 (live capture)

The total change splits into two contrasts on the same questions and judgments:
  k_fused 50 -> 100, everything else current:  flask_current_k100 minus replay_k50
  floor, fp16 and drift at k_fused=50:          replay_k50 minus previous_k50

Intervals reuse analyze_priority1.paired_delta_ci (10,000 resamples, seed 20260928, stratified by domain).
The two Priority 1 runs must reproduce Priority 1's own numbers here; the script stops if they do not.

Run from the worktree root, after every replay pair is judged:
    PYTHONPATH=src <python> <this file> [--datasets ip-200-fanout,crossdomain-60-fanout] [--dry]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
P1 = BASE.parent / "live-flask-current-2026-09-29-rootvenv"
sys.path.insert(0, str(P1))
sys.path.insert(0, str(BASE.parents[4] / "src"))

import analyze_priority1 as p1
from legal_rag.evaluation.graded_metrics import per_case_metrics, restrict_to_pool, summarize_runs
from legal_rag.evaluation.judgment_store import JudgmentStore
from legal_rag.evaluation.recall_report import prepare_evidence
from legal_rag.evaluation.relevance_judge import judge_key
from legal_rag.evaluation.relevance_pool import read_jsonl

PREVIOUS, REPLAY, CURRENT = "previous_k50", "replay_k50", "flask_current_k100"
CONTRASTS = {"k_fused_50_to_100": (REPLAY, CURRENT), "floor_fp16_drift_at_k50": (PREVIOUS, REPLAY),
             "total": (PREVIOUS, CURRENT)}
METRICS = ("success@5", "success@10", "precision@10_grade1", "precision@10_grade2", "ndcg@10", "mrr@10")
SHOWN = ("grade2_verified_quote_shown", "grade_ge1_document_shown")
# Priority 1 metrics that must reproduce exactly (nDCG depends on the pool, which now holds the replay pairs).
POOL_FREE = ("success@5", "success@10", "precision@10_grade1", "precision@10_grade2", "mrr@10")


def shown_metrics(ranked: list[dict], case_id: str, text_by_pair: dict, judgments: dict) -> dict[str, int]:
    """What Flask hands the model for one top 10: a verified grade-2 quote, and any grade >= 1 document."""
    evidence = prepare_evidence([{"chunk_id": h["chunk_id"], "document_id": h["document_id"],
                                  "chunk": {"text": text_by_pair[(case_id, h["chunk_id"])]}} for h in ranked[:10]])
    shown_text = [p1.squash(doc["text"]) for doc in evidence]
    shown_documents = {doc["document_id"] for doc in evidence}
    quote_shown = document_shown = 0
    for hit in ranked[:10]:
        judgment = judgments.get((case_id, hit["chunk_id"])) or {}
        if judgment.get("grade") not in (1, 2):
            continue
        document_shown = max(document_shown, int(hit["document_id"] in shown_documents))
        details = judgment.get("details") or {}
        quote = details.get("evidence_quote") or ""
        if judgment["grade"] == 2 and details.get("quote_verified") and quote:
            quote_shown = max(quote_shown, int(any(p1.squash(quote) in text for text in shown_text)))
    return dict(zip(SHOWN, (quote_shown, document_shown)))


def contrast(flat: dict, left: str, right: str, case_meta: dict, metric: str) -> dict:
    """right minus left. paired_delta_ci compares its two hard-coded slots, so present the pair under those names."""
    return p1.paired_delta_ci({p1.MODES[0]: flat[left], p1.MODES[1]: flat[right]}, case_meta, metric)


def analyze(name: str, key, case_meta: dict, priority1: dict) -> dict:
    pool_dir = BASE / "grading" / name
    pool_rows = read_jsonl(pool_dir / "pool.jsonl")
    store = JudgmentStore(pool_dir / "judgments.jsonl")
    grades, documents, failures, out_of_pool = restrict_to_pool(
        store.grades(key), store.documents(key), set(store.failures(key)),
        {(row["case_id"], row["chunk_id"]) for row in pool_rows})
    judgments = store.rows(key)
    runs = {path.stem: read_jsonl(path) for path in sorted((pool_dir / "runs").glob("*.jsonl"))}
    if set(runs) != {PREVIOUS, REPLAY, CURRENT}:
        raise SystemExit(f"{name}: unexpected runs {sorted(runs)}")
    missing = [(row["case_id"], hit["chunk_id"]) for mode in runs for row in runs[mode]
               for hit in row["ranked"][:10] if (row["case_id"], hit["chunk_id"]) not in grades]
    if missing or failures:
        raise SystemExit(f"{name}: {len(missing)} top-10 pairs lack a valid judgment, {len(failures)} failed parses")

    out = {"cases": len(runs[PREVIOUS]), "pool_pairs": len(pool_rows), "judgment_failures": len(failures),
           "out_of_pool_judgments_ignored": out_of_pool, "graded": {}, "model_visible": {}}
    flat_by_threshold = {}
    for threshold in (1, 2):
        summary = summarize_runs(runs, grades, documents, failures, depth=10, threshold=threshold)
        metrics = per_case_metrics(runs, grades, documents, failures, depth=10, threshold=threshold)
        flat = {mode: {row["case_id"]: row for row in metrics[mode]["chunk"]} for mode in runs}
        flat_by_threshold[threshold] = flat
        out["graded"][f"grade>={threshold}"] = {
            "means": {mode: {m: summary[mode]["chunk"].get(m) for m in METRICS} for mode in runs},
            "deltas": {label: {m: contrast(flat, left, right, case_meta, m) for m in METRICS}
                       for label, (left, right) in CONTRASTS.items()},
        }
        # Priority 1 regression: its two runs must come out the same on every pool-independent metric.
        reference = priority1["datasets"][name]["quality_by_threshold"][str(threshold)]["chunk_summary"]
        for mode in (PREVIOUS, CURRENT):
            for metric in POOL_FREE:
                got, want = summary[mode]["chunk"].get(metric), reference[mode][metric]
                if abs(got - want) > 1e-9:
                    raise SystemExit(f"{name} grade>={threshold} {mode} {metric}: {got} != Priority 1 {want}")

    text_by_pair = {(row["case_id"], row["chunk_id"]): row["text"] for row in pool_rows}
    shown = {mode: {row["case_id"]: shown_metrics(row["ranked"], row["case_id"], text_by_pair, judgments)
                    for row in runs[mode]} for mode in runs}
    reference = priority1["datasets"][name]["model_visible_evidence"]["summary"]
    for mode in (PREVIOUS, CURRENT):
        for metric in SHOWN:
            got = sum(r[metric] for r in shown[mode].values()) / len(shown[mode])
            if abs(got - reference[mode][metric]) > 1e-9:
                raise SystemExit(f"{name} {mode} {metric}: {got} != Priority 1 {reference[mode][metric]}")
    out["model_visible"] = {
        "means": {mode: {m: p1.avg([r[m] for r in shown[mode].values()]) for m in SHOWN} for mode in runs},
        "deltas": {label: {m: contrast(shown, left, right, case_meta, m) for m in SHOWN}
                   for label, (left, right) in CONTRASTS.items()},
    }
    return out


def table(name: str, result: dict) -> str:
    lines = [f"\n=== {name}: {result['cases']} cases, {result['pool_pairs']} pooled pairs, all top-10 pairs judged ==="]
    lines.append(f"{'metric':38s} {'2026-09-19':>10s} {'replay k50':>10s} {'current':>8s} | "
                 f"{'k_fused 50->100':>26s} | {'floor/fp16/drift @k50':>26s} | {'total':>26s}")

    def cell(d):
        return f"{d['delta']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]"

    def row(label, means, deltas, metric):
        return (f"{label:38s} {means[PREVIOUS][metric]:10.3f} {means[REPLAY][metric]:10.3f} {means[CURRENT][metric]:8.3f} | "
                f"{cell(deltas['k_fused_50_to_100'][metric]):>26s} | {cell(deltas['floor_fp16_drift_at_k50'][metric]):>26s} | "
                f"{cell(deltas['total'][metric]):>26s}")

    for grade, block in result["graded"].items():
        for metric in METRICS:
            if metric in ("precision@10_grade1", "precision@10_grade2", "ndcg@10") and grade != "grade>=1":
                continue
            lines.append(row(f"{metric} ({grade})" if "precision" not in metric and metric != "ndcg@10" else metric,
                             block["means"], block["deltas"], metric))
    for metric in SHOWN:
        lines.append(row(f"model-visible {metric}", result["model_visible"]["means"], result["model_visible"]["deltas"], metric))
    return "\n".join(lines)


def main() -> None:
    datasets = sys.argv[sys.argv.index("--datasets") + 1].split(",") if "--datasets" in sys.argv else [
        "ip-200-fanout", "crossdomain-60-fanout"]
    key = judge_key(p1.MODEL, think=True)
    if key.prompt_sha256 != p1.PROMPT_SHA256:
        raise SystemExit(f"unexpected judge prompt {key.prompt_sha256}")
    case_meta = p1.load_case_meta()
    priority1 = json.loads((P1 / "priority1-paired-analysis.json").read_text(encoding="utf-8"))
    results = {name: analyze(name, key, case_meta, priority1) for name in datasets}
    for name, result in results.items():
        print(table(name, result))
    if "--dry" in sys.argv:
        print("\n--dry: nothing written")
        return
    diagnostics = json.loads((BASE / "replay-diagnostics.json").read_text(encoding="utf-8"))["summary"]
    report = {"evaluation": "development_only",
              "judge": {"model": p1.MODEL, "prompt_sha256": key.prompt_sha256, "think": True,
                        "independent_human_calibration": False, "calibration_skipped": True},
              "bootstrap": {"replicates": p1.BOOTSTRAPS, "seed": p1.SEED, "stratified_by": "domain", "clustered_by": "group_id"},
              "datasets": results, "ranking_diagnostics": {name: diagnostics[name] for name in datasets}}
    with (BASE / "replay-analysis.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("\nwrote replay-analysis.json")


if __name__ == "__main__":
    main()

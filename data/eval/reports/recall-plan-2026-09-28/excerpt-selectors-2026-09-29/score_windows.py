#!/usr/bin/env python3
"""Score every candidate window of every top-10 document with the serving cross-encoder.

Reads no judgment. For each of the 260 development questions it rebuilds the per-document text Flask works from, cuts
it into windows (windows.py), and scores (query, window) pairs with the reranker as serving builds it (bge-reranker-v2-m3,
fp16, CUDA, max_length 512, batch 16). It also checks the two assumptions the scores rest on:

* the query in the Priority 1 target file is the query the captured reranker stage received (all 260 cases), and
* scoring a full chunk here reproduces the score the capture recorded for it (a sample), so the query, chunk text,
  precision and batching match serving.

Run from the worktree root, with the GPU free (no judge model loaded):
    PYTHONPATH=src <python> <this file> --check 6            # window statistics only, no GPU, nothing written
    PYTHONPATH=src <python> <this file> --check 6 --gpu      # also time the reranker on those cases, nothing written
    PYTHONPATH=src <python> <this file>                      # full run; writes window-scores.jsonl and score-run.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from functools import lru_cache
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent / "replay-kfused50-2026-09-29"))

import windows as wd  # noqa: E402

eb = wd.eb
MODEL = "BAAI/bge-reranker-v2-m3"
SEED = 20260929
POINTWISE_SAMPLE = {"ip-200-fanout": 10, "crossdomain-60-fanout": 5}


def load_cases() -> list[dict]:
    targets = {row["case_id"]: row for row in eb.read_jsonl(wd.LIVE / "target.jsonl")}
    cases = []
    for name in eb.DATASETS:
        pool_dir = wd.LIVE / "grading" / name
        pool = {(row["case_id"], row["chunk_id"]): row for row in eb.read_jsonl(pool_dir / "pool.jsonl")}
        for run in eb.read_jsonl(pool_dir / "runs" / "flask_current_k100.jsonl"):
            cases.append({"dataset": name, "case_id": run["case_id"], "query": targets[run["case_id"]]["query"],
                          "ranked": run["ranked"][:10], "pool": pool})
    return cases


def build_case(case: dict, count) -> list[dict]:
    docs = wd.doc_parts(case["case_id"], case["ranked"], case["pool"])
    reference = eb.build_case_docs(case["case_id"], case["ranked"], case["pool"])
    if [(d["document_id"], d["joined"]) for d in docs] != [(d["document_id"], d["joined"]) for d in reference]:
        raise ValueError(f"document grouping differs from the evidence-budget script for {case['case_id']}")
    wd.build_windows(docs, count(case["query"]), count)
    return docs


def window_stats(rows: list[dict]) -> dict:
    def summary(values):
        values = sorted(values)
        return {"min": values[0], "median": statistics.median(values), "p90": values[int(0.9 * (len(values) - 1))],
                "max": values[-1], "mean": round(statistics.mean(values), 1)}
    return {"cases": len(rows),
            "documents_per_case": summary([r["documents"] for r in rows]),
            "windows_per_case": summary([r["windows"] for r in rows]),
            "windows_per_case_top5_documents": summary([r["windows_top5"] for r in rows]),
            "window_chars": summary([c for r in rows for c in r["chars"]]),
            "pair_tokens": summary([t for r in rows for t in r["tokens"]]),
            "pairs_over_512_tokens": sum(t > wd.MAX_TOKENS for r in rows for t in r["tokens"]),
            "documents_no_longer_than_1200_chars": sum(r["short_docs"] for r in rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", type=int, default=0, help="only N sampled cases; write nothing")
    parser.add_argument("--gpu", action="store_true", help="with --check: also time the reranker")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)

    @lru_cache(maxsize=None)
    def count(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    cases = load_cases()
    if args.check:
        cases = random.Random(SEED).sample(cases, args.check)
    print(f"{len(cases)} cases", flush=True)

    # 1. windows and the exact reranker input length of every pair
    built, rows = [], []
    started = time.perf_counter()
    for case in cases:
        docs = build_case(case, count)
        texts = [(doc, span) for doc in docs for span in doc["windows"]]
        tokens = [len(tok(case["query"], doc["joined"][s:e])["input_ids"]) for doc, (s, e) in texts]
        rows.append({"documents": len(docs), "windows": len(texts), "windows_top5": sum(len(d["windows"]) for d in docs[:5]),
                     "chars": [e - s for _, (s, e) in texts], "tokens": tokens,
                     "short_docs": sum(len(d["joined"]) <= wd.CHAR_LIMIT for d in docs)})
        built.append(docs)
    print(f"windows built in {time.perf_counter() - started:.1f}s", flush=True)
    stats = window_stats(rows)
    print(json.dumps(stats, indent=1))
    if args.check and not args.gpu:
        return

    # 2. inputs the scores rest on: query equality (all cases) and recorded full-chunk scores (a sample)
    checks = {}
    if not args.check:
        import build_replay as br
        rng = random.Random(SEED)
        by_dataset = {}
        for case in cases:
            by_dataset.setdefault(case["dataset"], []).append(case["case_id"])
        sample = {cid for name, n in POINTWISE_SAMPLE.items() for cid in rng.sample(by_dataset[name], n)}
        queries = {case["case_id"]: case["query"] for case in cases}
        mismatched, recorded, seen = [], {}, 0
        print("streaming the capture for queries and recorded scores ...", flush=True)
        for record in br.stream_stages():
            case_id = record["case"]["case_id"]
            seen += 1
            if record["capture"]["query"] != queries[case_id]:
                mismatched.append(case_id)
            if case_id in sample:
                recorded[case_id] = br.RecordedReranker(record["capture"]).scores
        checks["query_equality"] = {"cases_streamed": seen, "mismatched": mismatched}
        if seen != len(cases) or mismatched:
            raise SystemExit(f"query check failed: streamed {seen}, mismatched {mismatched[:5]}")
        print(f"query check: {seen} cases, 0 mismatches", flush=True)

    # 3. the reranker exactly as serving builds it
    import torch
    from legal_rag.reranking.cross_encoder import CrossEncoderReranker
    reranker = CrossEncoderReranker(model=MODEL, device="cuda", dtype="float16", local_files_only=True)
    encoder = reranker.encoder()
    encoder.predict([(cases[0]["query"], "워밍업")] * 4, batch_size=16, show_progress_bar=False)
    torch.cuda.synchronize()

    def score(pairs):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        values = encoder.predict(pairs, batch_size=16, show_progress_bar=False)
        torch.cuda.synchronize()
        return [float(v) for v in values], time.perf_counter() - begin

    if not args.check:
        diffs, order_same = [], 0
        for case in cases:
            if case["case_id"] not in recorded:
                continue
            chunks = [(hit["chunk_id"], case["pool"][(case["case_id"], hit["chunk_id"])]["text"]) for hit in case["ranked"]]
            live, _ = score([(case["query"], text) for _, text in chunks])
            diffs.extend(abs(value - recorded[case["case_id"]][chunk_id]) for (chunk_id, _), value in zip(chunks, live))
        checks["full_chunk_scores_vs_recorded"] = {"cases": len(recorded), "pairs": len(diffs),
                                                  "max_abs_diff": max(diffs), "mean_abs_diff": statistics.mean(diffs)}
        print("full-chunk score check:", json.dumps(checks["full_chunk_scores_vs_recorded"]), flush=True)

    # 4. score every window
    out_rows, seconds = [], []
    for case, docs in zip(cases, built):
        pairs = [(case["query"], doc["joined"][s:e]) for doc in docs for s, e in doc["windows"]]
        values, elapsed = score(pairs)
        seconds.append(elapsed)
        cursor, doc_rows = 0, []
        for doc in docs:
            window_rows = []
            for s, e in doc["windows"]:
                window_rows.append({"start": s, "end": e, "score": values[cursor]})
                cursor += 1
            doc_rows.append({"document_id": doc["document_id"], "rank": doc["rank"], "joined_chars": len(doc["joined"]),
                             "windows": window_rows})
        out_rows.append({"dataset": case["dataset"], "case_id": case["case_id"], "query": case["query"],
                         "pairs": len(pairs), "score_seconds": elapsed, "docs": doc_rows})
        print(f"{case['case_id'][:16]} {case['dataset'][:5]} {len(pairs):3d} pairs {elapsed:5.2f}s", flush=True)
    timing = {"cases": len(seconds), "mean_pairs": statistics.mean(r["pairs"] for r in out_rows),
              "mean_seconds": statistics.mean(seconds), "median_seconds": statistics.median(seconds),
              "p90_seconds": sorted(seconds)[int(0.9 * (len(seconds) - 1))], "max_seconds": max(seconds),
              "seconds_per_pair_ms": 1000 * sum(seconds) / sum(r["pairs"] for r in out_rows),
              "gpu": torch.cuda.get_device_name(0), "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20}
    print(json.dumps(timing, indent=1))
    if args.check:
        return

    import hashlib
    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    with (BASE / "window-scores.jsonl").open("x", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"reranker": {"model": MODEL, "device": "cuda", "dtype": "float16", "max_length": 512, "batch_size": 16},
                "window_stats": stats, "timing": timing, "checks": checks,
                "sha256": {"windows.py": sha(BASE / "windows.py"), "score_windows.py": sha(Path(__file__)),
                           "window-scores.jsonl": sha(BASE / "window-scores.jsonl"),
                           "target.jsonl": sha(wd.LIVE / "target.jsonl")}}
    (BASE / "score-run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("wrote window-scores.jsonl and score-run.json")


if __name__ == "__main__":
    main()

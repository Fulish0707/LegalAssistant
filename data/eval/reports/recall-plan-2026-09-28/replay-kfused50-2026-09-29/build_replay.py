#!/usr/bin/env python3
"""Offline k_fused=50 replay of the Priority 1 capture: build the replay run and its judging pools.

Each captured case holds all eight legs at depth 100 and the cross-encoder score of every fused candidate.
Re-fusing the legs at k_fused=50 (leg_floor=3; weights and RRF constant as served) gives the candidate set a
k_fused=50 search would have sent to the reranker. Reranking is pointwise, so those candidates keep the recorded
fp16 scores; the order matches CrossEncoderReranker (score descending, ties by fused order). Nothing is retrieved or
reranked again, and no serving setting changes.

Run from the worktree root:
    PYTHONPATH=src <python> <this file> --check 3   # replay the first 3 cases, print, write nothing
    PYTHONPATH=src <python> <this file>             # replay all cases; write manifest, pools and judge queues
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
P1 = BASE.parent / "live-flask-current-2026-09-29-rootvenv"
sys.path.insert(0, str(BASE.parents[4] / "src"))

from legal_rag.evaluation.judgment_store import JudgmentStore
from legal_rag.evaluation.recall_stages import replay_stages
from legal_rag.evaluation.recall_variants import BASELINE
from legal_rag.evaluation.relevance_judge import judge_key
from legal_rag.evaluation.relevance_pool import _pair, adopt_run, read_jsonl, write_pool
from legal_rag.schemas import Chunk

K50 = replace(BASELINE, name="K50", budget=50)
DATASETS = ("ip-200-fanout", "crossdomain-60-fanout")
MODEL, THINK = "ollama:qwen3:8b", True
PREVIOUS, CURRENT, REPLAY = "previous_k50", "flask_current_k100", "replay_k50"
POOL_DEPTH = 10


class RecordedReranker:
    """Replays the scores recorded in a capture's reranker stage.

    Scoring is pointwise (each query and chunk text is scored alone), so a smaller candidate set keeps its scores.
    Ordering matches CrossEncoderReranker: score descending, ties by input order. A candidate without a recorded
    score raises, so a replay can never invent one.
    """

    name = "recorded"

    def __init__(self, capture: dict) -> None:
        stage = capture["reranked"]
        if stage["error"] is not None:
            raise ValueError(f"captured rerank stage failed: {stage['error']}")
        self.scores = {hit["chunk_id"]: hit["score"] for hit in stage["hits"]}

    def rerank(self, query, candidates, k):
        missing = [c.chunk.id for c in candidates if c.chunk.id not in self.scores]
        if missing:
            raise KeyError(f"{len(missing)} candidates have no recorded score, e.g. {missing[0]}")
        order = sorted(range(len(candidates)), key=lambda i: (-self.scores[candidates[i].chunk.id], i))
        return [candidates[i].model_copy(update={"score": self.scores[candidates[i].chunk.id], "retriever": "reranked"})
                for i in order[:k]]


def ids(stage: dict) -> list[str]:
    return [hit["chunk_id"] for hit in stage["hits"]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=BASE.parents[4]).stdout.strip()


def replay_case(capture: dict, current_top10: list[str], previous_top10: list[str]) -> tuple[dict, list[dict]]:
    """Replay one case at k_fused=100 (parity check) and at k_fused=50; return diagnostics and the K50 top-10 hits."""
    reranker = RecordedReranker(capture)
    base = replay_stages(capture, reranker, variant=BASELINE)
    k50 = replay_stages(capture, reranker, variant=K50)
    fused100, fused50 = set(ids(capture["fused"])), ids(k50["fused"])
    top10_100, top10_50 = ids(capture["reranked"])[:10], ids(k50["reranked"])[:10]
    corpus = {hit["chunk_id"]: leg["name"].rsplit(":", 1)[0] for leg in capture["legs"] for hit in leg["hits"]}
    diagnostics = {
        "parity_fused": ids(base["fused"]) == ids(capture["fused"]),
        "parity_reranked": (ids(base["reranked"]) == ids(capture["reranked"])
                            and [h["score"] for h in base["reranked"]["hits"]] == [h["score"] for h in capture["reranked"]["hits"]]),
        "parity_run_file": top10_100 == current_top10,
        "capture_rerank_scored_every_candidate": len(capture["reranked"]["hits"]) == len(capture["fused"]["hits"]),
        "k50_candidates": len(fused50),
        "k50_subset_of_k100": set(fused50) <= fused100,
        "current_top10_outside_k50_candidates": sum(1 for c in top10_100 if c not in set(fused50)),
        "overlap_replay_current": len(set(top10_50) & set(top10_100)),
        "overlap_replay_previous": len(set(top10_50) & set(previous_top10)),
        "corpus_share_replay": Counter(corpus.get(c, "unknown") for c in top10_50),
        "corpus_share_current": Counter(corpus.get(c, "unknown") for c in top10_100),
        "corpus_share_previous": Counter(corpus.get(c, "unknown") for c in previous_top10),
    }
    return diagnostics, k50["reranked"]["hits"][:POOL_DEPTH]


def stream_stages(limit: int | None = None):
    with (P1 / "stages.jsonl").open(encoding="utf-8") as handle:
        for count, line in enumerate(handle):
            if limit is not None and count >= limit:
                return
            if line.strip():
                yield json.loads(line)


def main() -> None:
    limit = int(sys.argv[sys.argv.index("--check") + 1]) if "--check" in sys.argv else None
    key = judge_key(MODEL, think=THINK)
    grading = {ds: {"pool": read_jsonl(P1 / "grading" / ds / "pool.jsonl"),
                    "runs": {m: read_jsonl(P1 / "grading" / ds / "runs" / f"{m}.jsonl") for m in (PREVIOUS, CURRENT)}}
               for ds in DATASETS}
    for info in grading.values():
        info["have"] = {(r["case_id"], r["chunk_id"]) for r in info["pool"]}
        info["question"] = {r["case_id"]: r["question"] for r in info["pool"]}
        info["answer"] = {r["case_id"]: r["reference_answer"] for r in info["pool"]}
        info["previous"] = {r["case_id"]: [h["chunk_id"] for h in r["ranked"][:POOL_DEPTH]] for r in info["runs"][PREVIOUS]}
        info["current"] = {r["case_id"]: [h["chunk_id"] for h in r["ranked"][:POOL_DEPTH]] for r in info["runs"][CURRENT]}
    replay_rows = defaultdict(dict)       # dataset -> case_id -> run row
    payloads = defaultdict(dict)          # dataset -> (case_id, chunk_id) -> chunk payload, for pairs the pool lacks
    relevant = {}                         # case_id -> reference document ids
    per_case = defaultdict(dict)          # dataset -> case_id -> diagnostics

    for record in stream_stages(limit):
        case, capture = record["case"], record["capture"]
        ds, case_id = case["dataset"], case["case_id"]
        info = grading[ds]
        relevant[case_id] = case.get("relevant_document_ids") or []
        diagnostics, top10 = replay_case(capture, info["current"][case_id], info["previous"][case_id])
        per_case[ds][case_id] = diagnostics
        replay_rows[ds][case_id] = {"case_id": case_id, "ranked": [
            {"chunk_id": h["chunk_id"], "document_id": h["document_id"]} for h in top10]}
        for h in top10:
            if (case_id, h["chunk_id"]) not in info["have"]:
                payloads[ds][(case_id, h["chunk_id"])] = h["chunk"]

    checks = ("parity_fused", "parity_reranked", "parity_run_file", "capture_rerank_scored_every_candidate", "k50_subset_of_k100")
    summary = {}
    for ds, cases in per_case.items():
        n = len(cases)
        mean = lambda name: sum(d[name] for d in cases.values()) / n
        share = lambda name: {c: sum(d[name][c] for d in cases.values()) / (10 * n)
                              for c in sorted({c for d in cases.values() for c in d[name]})}
        summary[ds] = {
            "cases": n,
            **{f"{c}_cases": sum(bool(d[c]) for d in cases.values()) for c in checks},
            "k50_candidates_min_max": [min(d["k50_candidates"] for d in cases.values()), max(d["k50_candidates"] for d in cases.values())],
            "mean_current_top10_outside_k50_candidates": mean("current_top10_outside_k50_candidates"),
            "mean_top10_overlap_replay_vs_current": mean("overlap_replay_current"),
            "mean_top10_overlap_replay_vs_previous": mean("overlap_replay_previous"),
            "corpus_share_of_top10": {"previous_k50": share("corpus_share_previous"), "replay_k50": share("corpus_share_replay"),
                                      "flask_current_k100": share("corpus_share_current")},
            "pairs_not_in_priority1_pool": len(payloads[ds]),
        }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    failed = {ds: {c: s[f"{c}_cases"] for c in checks if s[f"{c}_cases"] != s["cases"]} for ds, s in summary.items()}
    if any(failed.values()):
        sys.exit(f"parity/subset checks failed, nothing written: {failed}")
    if limit is not None:
        print(f"--check {limit}: all checks passed; nothing written")
        return

    started = datetime.now(timezone.utc).isoformat()
    outputs = {}
    for ds in DATASETS:
        info = grading[ds]
        rows = [replay_rows[ds][r["case_id"]] for r in info["runs"][PREVIOUS]]     # same case order as the other runs
        extra = []
        for (case_id, chunk_id), payload in payloads[ds].items():
            chunk = Chunk.model_validate(payload)
            row = _pair({"case_id": case_id, "question": info["question"][case_id], "answer": info["answer"][case_id],
                         "document_id": ""}, chunk)
            row["is_gold_chunk"] = chunk.document_id in relevant[case_id]
            extra.append(row)
        pool, runs = adopt_run(info["pool"], info["runs"], REPLAY, rows, extra, pool_depth=POOL_DEPTH)
        meta = json.loads((P1 / "grading" / ds / "pool.json").read_text(encoding="utf-8"))
        for stale in ("pairs", "modes"):
            meta.pop(stale)
        meta.update(experiment="replay_kfused50_2026_09_29", settings_hash="replay-k50-floor3-recorded-fp16-scores",
                    replay={"variant": asdict(K50), "candidates": "eight captured legs re-fused at k_fused=50",
                            "scores": "cross-encoder fp16 scores recorded in the k_fused=100 capture (pointwise)",
                            "runs": {PREVIOUS: "2026-09-19 fan-out: k_fused=50, no floor, fp32 (recorded earlier)",
                                     CURRENT: "current served configuration: k_fused=100, floor 3, fp16 (live capture)",
                                     REPLAY: "current configuration re-fused at k_fused=50, floor 3, fp16 scores replayed"}})
        pool_dir = write_pool(BASE / "grading" / ds, pool, runs, meta)
        shutil.copy2(P1 / "grading" / ds / "judgments.jsonl", pool_dir / "judgments.jsonl")
        store = JudgmentStore(pool_dir / "judgments.jsonl")
        queue = [r for r in pool if not store.has(key, r["case_id"], r["chunk_id"])]
        with (pool_dir / "judge-queue.jsonl").open("x", encoding="utf-8") as handle:
            for r in queue:
                handle.write(json.dumps(r, ensure_ascii=False) + "\n")
        outputs[ds] = {"pool_dir": str(pool_dir.relative_to(BASE)), "pool_pairs": len(pool), "queued_for_judging": len(queue),
                       "judged_before": len(pool) - len(queue)}
        summary[ds].update(outputs[ds])

    manifest = {
        "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "development-only offline replay: separate the k_fused effect from floor/fp16/time drift in Priority 1",
        "script": "build_replay.py", "script_sha256": sha256(Path(__file__)),
        "git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
        "variant_replayed": asdict(K50), "parity_variant": asdict(BASELINE),
        "judge": {"model": MODEL, "prompt_sha256": key.prompt_sha256, "think": THINK},
        "inputs_sha256": {"stages.jsonl": sha256(P1 / "stages.jsonl"), "target.jsonl": sha256(P1 / "target.jsonl"),
                          **{f"grading/{ds}/{name}": sha256(P1 / "grading" / ds / name)
                             for ds in DATASETS for name in ("pool.jsonl", "judgments.jsonl")}},
        "outputs": outputs,
    }
    (BASE / "manifest.json").open("x", encoding="utf-8").write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (BASE / "replay-diagnostics.json").open("x", encoding="utf-8").write(
        json.dumps({"summary": summary, "per_case": per_case}, ensure_ascii=False, indent=1) + "\n")
    print("\nwrote manifest.json, replay-diagnostics.json and grading/*")


if __name__ == "__main__":
    main()

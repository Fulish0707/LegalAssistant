#!/usr/bin/env python3
"""Reproduce the paired Priority 1 retrieval and Flask evidence analysis."""
from __future__ import annotations

import hashlib
import json
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[5]
sys.path.insert(0, str(ROOT / "src"))

from legal_rag.evaluation.graded_metrics import per_case_metrics, restrict_to_pool, summarize_runs
from legal_rag.evaluation.judgment_store import JudgmentStore
from legal_rag.evaluation.recall_report import prepare_evidence
from legal_rag.evaluation.relevance_judge import judge_key
from legal_rag.evaluation.relevance_pool import read_jsonl

MODEL = "ollama:qwen3:8b"
PROMPT_SHA256 = "7a2b826ac9cb93494241c98ccc829755d2af0ee37ebd8e3178cdbd798ecb596c"
BOOTSTRAPS = 10_000
SEED = 20260928
MODES = ("previous_k50", "flask_current_k100")
METRICS = ("success@5", "success@10", "precision@10", "precision@10_grade1",
           "precision@10_grade2", "ndcg@10", "mrr@10")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def avg(values: list[float | int | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return statistics.mean(valid) if valid else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * q + 0.5)))
    return ordered[index]


def squash(text: str) -> str:
    """Collapse whitespace runs. The judge's evidence quotes do; chunk text keeps double spaces and newlines."""
    return re.sub(r"\s+", " ", text).strip()


def paired_delta_ci(per_case: dict[str, dict[str, dict]], case_meta: dict[str, dict],
                    metric: str) -> dict:
    deltas = {}
    for case_id in case_meta:
        left = per_case[MODES[0]].get(case_id, {}).get(metric)
        right = per_case[MODES[1]].get(case_id, {}).get(metric)
        if left is not None and right is not None:
            deltas[case_id] = float(right) - float(left)

    groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for case_id, meta in case_meta.items():
        if case_id in deltas:
            groups[meta["domain"]][meta["group_id"]].append(case_id)
    domains = sorted(groups)
    if not domains:
        return {"delta": None, "ci95": [None, None], "cases": 0}

    point_by_domain = {
        domain: avg([deltas[case_id] for group_cases in groups[domain].values()
                     for case_id in group_cases])
        for domain in domains
    }
    point = statistics.mean(point_by_domain.values())
    rng = random.Random(SEED)
    draws = []
    for _ in range(BOOTSTRAPS):
        domain_means = []
        for domain in domains:
            cluster_ids = list(groups[domain])
            sampled = [rng.choice(cluster_ids) for _ in cluster_ids]
            values = [deltas[case_id] for cluster in sampled for case_id in groups[domain][cluster]]
            domain_means.append(statistics.mean(values))
        draws.append(statistics.mean(domain_means))
    return {"delta": point, "ci95": [percentile(draws, .025), percentile(draws, .975)],
            "cases": len(deltas), "domains": point_by_domain, "bootstrap_replicates": BOOTSTRAPS,
            "bootstrap_seed": SEED, "clustered_by": "group_id", "stratified_by": "domain"}


def load_case_meta() -> dict[str, dict]:
    return {row["case_id"]: {"domain": row["domain"], "group_id": row["group_id"]}
            for row in jsonl(BASE / "per-case.jsonl")}


def write_judge_provenance(name: str, key) -> dict:
    pool_dir = BASE / "grading" / name
    manifest = json.loads((pool_dir / "manifest.json").read_text(encoding="utf-8"))
    capture = json.loads((BASE / "manifest.json").read_text(encoding="utf-8"))
    model = next((row for row in capture.get("ollama_models", []) if row.get("name") == "qwen3:8b"), {})
    queue = jsonl(pool_dir / "judge-queue.jsonl")
    history = jsonl(pool_dir / "judgments.jsonl")
    queued_pairs = {(row["case_id"], row["chunk_id"]) for row in queue}
    latest = JudgmentStore(pool_dir / "judgments.jsonl").rows(key)
    covered = [latest[pair] for pair in queued_pairs if pair in latest]
    recovered = []
    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in history:
        pair = (row.get("case_id"), row.get("chunk_id"))
        if pair in queued_pairs:
            by_pair[pair].append(row)
    for pair, rows in by_pair.items():
        if len(rows) > 1 and rows[-1].get("grade") in (0, 1, 2) and any(row.get("grade") is None for row in rows[:-1]):
            recovered.append({"case_id": pair[0], "chunk_id": pair[1]})
    provenance = {
        "purpose": "development-only paired top-10 retrieval comparison",
        "judge_model": key.model,
        "model_digest": model.get("digest"),
        "prompt_sha256": key.prompt_sha256,
        "think": True,
        "max_tokens": 4096,
        "num_ctx": 8192,
        "calibration_skipped": True,
        "independent_human_calibration": False,
        "calibration_reason": "No independent human labels are available; this does not pass the plan's human-calibration gate.",
        "queue_pairs": len(queue),
        "queue_final_valid": sum(row.get("grade") in (0, 1, 2) for row in covered),
        "queue_final_failed": sum(row.get("grade") is None for row in covered),
        "recovered_parse_failures": recovered,
        "judged_at_utc": {
            "first": min((row.get("judged_at", "") for row in covered), default=None),
            "last": max((row.get("judged_at", "") for row in covered), default=None),
        },
        "seeded_valid_labels": manifest.get("valid_existing_labels_seeded"),
        "source_pool_sha256": manifest.get("source_pool_sha256"),
        "queue_sha256": sha256(pool_dir / "judge-queue.jsonl"),
        "judgments_sha256": sha256(pool_dir / "judgments.jsonl"),
        "commands": [
            "PYTHONPATH=src python scripts/judge_relevance.py --pool-dir <pool> --pool-file <pool>/judge-queue.jsonl --skip-calibration-check --think",
            "IP resume additionally used --retry-failed after preserving the interrupted judgments file and removing its 176 trailing NUL bytes.",
        ] if name == "ip-200-fanout" else [
            "PYTHONPATH=src python scripts/judge_relevance.py --pool-dir <pool> --pool-file <pool>/judge-queue.jsonl --skip-calibration-check --think"
        ],
    }
    path = pool_dir / "judge-run-provenance.json"
    path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"file": str(path.relative_to(BASE)), "sha256": sha256(path)}


def run_dataset(name: str, key) -> dict:
    pool_dir = BASE / "grading" / name
    pool_rows = read_jsonl(pool_dir / "pool.jsonl")
    pool_pairs = {(row["case_id"], row["chunk_id"]) for row in pool_rows}
    store = JudgmentStore(pool_dir / "judgments.jsonl")
    grades, documents, failures, out_of_pool = restrict_to_pool(
        store.grades(key), store.documents(key), set(store.failures(key)), pool_pairs)
    judgments = store.rows(key)
    runs = {path.stem: read_jsonl(path) for path in sorted((pool_dir / "runs").glob("*.jsonl"))}
    if tuple(mode for mode in MODES if mode in runs) != MODES:
        raise ValueError(f"{name}: expected runs {MODES}; found {tuple(runs)}")

    queue = jsonl(pool_dir / "judge-queue.jsonl")
    queue_coverage = sum((row["case_id"], row["chunk_id"]) in grades for row in queue)
    if queue_coverage != len(queue):
        raise ValueError(f"{name}: valid judge coverage {queue_coverage}/{len(queue)}")

    case_meta = load_case_meta()
    target = {row["case_id"]: row for row in jsonl(BASE / "target.jsonl")}
    data = {"case_count": len(target), "judge_queue": {"pairs": len(queue), "valid": queue_coverage},
            "judgment_failures_in_pool": len(failures), "out_of_pool_judgments_ignored": out_of_pool,
            "settings": json.loads((pool_dir / "pool.json").read_text(encoding="utf-8"))["retrieval"],
            "quality_by_threshold": {}, "known_source_hit_at_k": {}, "top10_overlap": {}}
    data["case_count"] = len(runs[MODES[0]])
    data["artifacts_sha256"] = {
        path.relative_to(pool_dir).as_posix(): sha256(path)
        for path in (pool_dir / "pool.json", pool_dir / "pool.jsonl", pool_dir / "manifest.json",
                     pool_dir / "judge-queue.jsonl", pool_dir / "judgments.jsonl",
                     *(sorted((pool_dir / "runs").glob("*.jsonl"))))
    }

    per_case_at_threshold = {}
    for threshold in (1, 2):
        summary = summarize_runs(runs, grades, documents, failures, depth=10, threshold=threshold)
        metrics = per_case_metrics(runs, grades, documents, failures, depth=10, threshold=threshold)
        flat = {mode: {row["case_id"]: row for row in metrics[mode]["chunk"]} for mode in MODES}
        per_case_at_threshold[threshold] = flat
        data["quality_by_threshold"][str(threshold)] = {
            "headline_relevance": f"grade >= {threshold}",
            "chunk_summary": {mode: {metric: summary[mode]["chunk"].get(metric) for metric in METRICS}
                              for mode in MODES},
            "current_minus_previous": {metric: paired_delta_ci(flat, case_meta, metric) for metric in METRICS},
            "by_domain": {
                domain: {
                    "chunk_summary": {
                        mode: {metric: avg([row.get(metric) for case_id, row in flat[mode].items()
                                            if case_meta.get(case_id, {}).get("domain") == domain])
                               for metric in METRICS}
                        for mode in MODES
                    },
                    "current_minus_previous": {
                        metric: paired_delta_ci(
                            flat, {case_id: meta for case_id, meta in case_meta.items()
                                   if meta["domain"] == domain}, metric)
                        for metric in METRICS
                    },
                }
                for domain in sorted({case_meta[case_id]["domain"] for case_id in flat[MODES[0]]})
            },
        }

    for mode in MODES:
        available_depth = min((len(row["ranked"]) for row in runs[mode]), default=0)
        hit_rates = {"run_depth": available_depth, "rates": {}}
        for cutoff in (5, 10, 20, 50, 100):
            if cutoff > available_depth:
                continue
            hits = []
            for row in runs[mode]:
                refs = set(target[row["case_id"]].get("relevant_document_ids") or [])
                hits.append(int(any(item.get("document_id") in refs for item in row["ranked"][:cutoff])))
            hit_rates["rates"][str(cutoff)] = avg(hits)
        data["known_source_hit_at_k"][mode] = hit_rates

    old = {row["case_id"]: [item["chunk_id"] for item in row["ranked"][:10]] for row in runs[MODES[0]]}
    cur = {row["case_id"]: [item["chunk_id"] for item in row["ranked"][:10]] for row in runs[MODES[1]]}
    overlaps = [len(set(old[c]) & set(cur[c])) for c in old]
    data["top10_overlap"] = {
        "mean_shared_chunks": avg(overlaps), "mean_shared_slots_fraction": avg([x / 10 for x in overlaps]),
        "ordered_top10_identical_cases": sum(old[c] == cur[c] for c in old),
        "cases": len(old), "new_current_slots_vs_previous": sum(10 - x for x in overlaps),
    }

    current_case_ids = set(target)
    current_capture = {row["case_id"]: row for row in jsonl(BASE / "per-case.jsonl")
                       if row["case_id"] in current_case_ids}
    domains_in_dataset = sorted({row["domain"] for row in target.values()})
    data["current_stage_known_source_metrics"] = {
        "by_domain": {
            domain: {
                stage: {metric: avg([row["reference_metrics"][stage].get(metric)
                                     for row in current_capture.values() if row["domain"] == domain])
                        for metric in ("reference_coverage", "chunk_cutoff_recall", "unique_document_recall")
                        if any(metric in row["reference_metrics"][stage]
                               for row in current_capture.values() if row["domain"] == domain)}
                for stage in ("union", "fused", "top10", "prepared")
            }
            for domain in domains_in_dataset
        }
    }

    # Qwen-quoted grade-2 spans are checked against the text actually passed through Flask's
    # five-document / 1,200-character evidence formatter. Whitespace runs are collapsed on both
    # sides: the judge's quote collapses the double spaces the source keeps, so an exact substring
    # test reads visible quotes as lost. The exact-match counts stay for comparison.
    current_run = {row["case_id"]: row["ranked"][:10] for row in runs[MODES[1]]}
    visible_rows = {row["case_id"]: row for row in current_capture.values()}
    visible_by_pair: dict[tuple[str, str], str] = {}
    for case_id, row in visible_rows.items():
        for document in row["evidence"]:
            for segment in document["segments"]:
                text = document["text"][segment["evidence_start"]:segment["evidence_end"]]
                pair = (case_id, segment["chunk_id"])
                visible_by_pair[pair] = visible_by_pair.get(pair, "") + text
    counts = {"top10_chunks": 0, "unjudged_top10_chunks": 0, "grade_ge1_chunks": 0,
              "grade2_chunks": 0, "grade2_verified_quotes": 0, "grade_ge1_chunks_visible": 0,
              "grade2_chunks_with_any_text_visible": 0,
              "grade2_quotes_visible_exact_match": 0, "grade2_quotes_visible_own_excerpt": 0,
              "grade2_quotes_visible_elsewhere_in_evidence": 0,
              "grade2_quotes_lost_document_cap": 0,
              "grade2_quotes_lost_chunk_not_in_excerpt": 0, "grade2_quotes_lost_past_excerpt_cut": 0,
              "cases_grade_ge1_success": 0,
              "cases_grade_ge1_success_with_relevant_text_visible": 0,
              "cases_grade2_success": 0, "cases_grade2_success_with_quote_visible_exact_match": 0,
              "cases_grade2_success_with_quote_visible_own_excerpt": 0,
              "cases_grade2_success_with_quote_visible_anywhere": 0}
    for case_id, ranked in current_run.items():
        relevant_visible = False
        grade1_success = False
        grade2_success = False
        quote_visible_exact = quote_visible_own = quote_visible_anywhere = False
        selected_documents = {doc["document_id"] for doc in visible_rows[case_id]["evidence"]}
        shown_text = [squash(doc["text"]) for doc in visible_rows[case_id]["evidence"]]
        for hit in ranked:
            counts["top10_chunks"] += 1
            pair = (case_id, hit["chunk_id"])
            judgment = judgments.get(pair)
            if judgment is None or judgment.get("grade") not in (0, 1, 2):
                counts["unjudged_top10_chunks"] += 1
                continue
            grade = judgment["grade"]
            excerpt = visible_by_pair.get(pair, "")
            if grade >= 1:
                counts["grade_ge1_chunks"] += 1
                grade1_success = True
                if excerpt.strip():
                    counts["grade_ge1_chunks_visible"] += 1
                    relevant_visible = True
            if grade == 2:
                counts["grade2_chunks"] += 1
                grade2_success = True
                if excerpt.strip():
                    counts["grade2_chunks_with_any_text_visible"] += 1
                details = judgment.get("details") or {}
                quote = details.get("evidence_quote") or ""
                if details.get("quote_verified") and quote:
                    counts["grade2_verified_quotes"] += 1
                    if quote in excerpt:
                        counts["grade2_quotes_visible_exact_match"] += 1
                        quote_visible_exact = True
                    squashed = squash(quote)
                    if squashed in squash(excerpt):
                        counts["grade2_quotes_visible_own_excerpt"] += 1
                        quote_visible_own = quote_visible_anywhere = True
                    elif any(squashed in text for text in shown_text):
                        counts["grade2_quotes_visible_elsewhere_in_evidence"] += 1
                        quote_visible_anywhere = True
                    elif hit.get("document_id") not in selected_documents:
                        counts["grade2_quotes_lost_document_cap"] += 1
                    elif pair not in visible_by_pair:
                        counts["grade2_quotes_lost_chunk_not_in_excerpt"] += 1
                    else:
                        counts["grade2_quotes_lost_past_excerpt_cut"] += 1
        counts["cases_grade_ge1_success"] += int(grade1_success)
        counts["cases_grade_ge1_success_with_relevant_text_visible"] += int(grade1_success and relevant_visible)
        counts["cases_grade2_success"] += int(grade2_success)
        counts["cases_grade2_success_with_quote_visible_exact_match"] += int(grade2_success and quote_visible_exact)
        counts["cases_grade2_success_with_quote_visible_own_excerpt"] += int(grade2_success and quote_visible_own)
        counts["cases_grade2_success_with_quote_visible_anywhere"] += int(grade2_success and quote_visible_anywhere)
    cases = len(current_run)
    counts["grade2_quotes_lost_within_selected_document"] = (
        counts["grade2_quotes_lost_chunk_not_in_excerpt"] + counts["grade2_quotes_lost_past_excerpt_cut"])
    counts["grade_ge1_chunk_text_retention_rate"] = (
        counts["grade_ge1_chunks_visible"] / counts["grade_ge1_chunks"] if counts["grade_ge1_chunks"] else None)
    verified = counts["grade2_verified_quotes"]
    counts["grade2_quote_retention_rate_exact_match"] = (
        counts["grade2_quotes_visible_exact_match"] / verified if verified else None)
    counts["grade2_quote_retention_rate_own_excerpt"] = (
        counts["grade2_quotes_visible_own_excerpt"] / verified if verified else None)
    counts["grade2_quote_retention_rate_anywhere"] = (
        (counts["grade2_quotes_visible_own_excerpt"] + counts["grade2_quotes_visible_elsewhere_in_evidence"])
        / verified if verified else None)
    counts["grade_ge1_case_success"] = counts["cases_grade_ge1_success"] / cases
    counts["grade_ge1_success_with_visible_relevant_text"] = (
        counts["cases_grade_ge1_success_with_relevant_text_visible"] / cases)
    counts["grade2_case_success"] = counts["cases_grade2_success"] / cases
    counts["grade2_success_with_verified_quote_visible_exact_match"] = (
        counts["cases_grade2_success_with_quote_visible_exact_match"] / cases)
    counts["grade2_success_with_verified_quote_visible_own_excerpt"] = (
        counts["cases_grade2_success_with_quote_visible_own_excerpt"] / cases)
    counts["grade2_success_with_verified_quote_visible_anywhere"] = (
        counts["cases_grade2_success_with_quote_visible_anywhere"] / cases)
    parity = json.loads((BASE / "flask-evidence-parity.json").read_text(encoding="utf-8"))
    counts["flask_formatter_parity"] = parity
    data["current_flask_evidence_survival"] = counts

    # What the model is shown, for both runs: Flask's formatter applied to each run's top 10, scored
    # with the same verified grade-2 quotes. The previous run recorded no evidence, so it is rebuilt
    # from the pool text; the current run's rebuild must equal its captured evidence.
    text_by_pair = {(row["case_id"], row["chunk_id"]): row["text"] for row in pool_rows}
    shown: dict[str, dict[str, dict[str, int]]] = {mode: {} for mode in MODES}
    for mode in MODES:
        for row in runs[mode]:
            case_id, ranked = row["case_id"], row["ranked"][:10]
            evidence = prepare_evidence([
                {"chunk_id": hit["chunk_id"], "document_id": hit["document_id"],
                 "chunk": {"text": text_by_pair[(case_id, hit["chunk_id"])]}} for hit in ranked])
            if mode == MODES[1]:
                captured = [(doc["document_id"], doc["text"]) for doc in visible_rows[case_id]["evidence"]]
                if [(doc["document_id"], doc["text"]) for doc in evidence] != captured:
                    raise ValueError(f"{name}: {case_id}: rebuilt evidence differs from the captured evidence")
            shown_text = [squash(doc["text"]) for doc in evidence]
            shown_documents = {doc["document_id"] for doc in evidence}
            quote_shown = document_shown = 0
            for hit in ranked:
                judgment = judgments.get((case_id, hit["chunk_id"])) or {}
                if judgment.get("grade") not in (1, 2):
                    continue
                document_shown = max(document_shown, int(hit["document_id"] in shown_documents))
                details = judgment.get("details") or {}
                quote = details.get("evidence_quote") or ""
                if judgment["grade"] == 2 and details.get("quote_verified") and quote:
                    quote_shown = max(quote_shown, int(any(squash(quote) in text for text in shown_text)))
            shown[mode][case_id] = {"grade2_verified_quote_shown": quote_shown,
                                    "grade_ge1_document_shown": document_shown}
    shown_metrics = ("grade2_verified_quote_shown", "grade_ge1_document_shown")
    data["model_visible_evidence"] = {
        "scope": "Evidence from the forced pre-search. _with_legal_evidence (langchain_config.py) removes "
                 "search_legal_corpus from the tool list when the pre-search returned text, so this is the "
                 "legal text the model has for such a question.",
        "summary": {mode: {metric: avg([row[metric] for row in shown[mode].values()])
                           for metric in shown_metrics} for mode in MODES},
        "current_minus_previous": {metric: paired_delta_ci(shown, case_meta, metric)
                                   for metric in shown_metrics},
    }
    return data


def stage_timings() -> dict:
    path = BASE / "stages.jsonl"
    legs: dict[str, list[float]] = defaultdict(list)
    fusion, rerank = [], []
    record_count = 0
    with path.open(encoding="utf-8") as handle:
        rows = (json.loads(line) for line in handle if line.strip())
        for row in rows:
            record_count += 1
            capture = row["capture"]
            for leg in capture["legs"]:
                if leg.get("timing_ms") is not None:
                    legs[leg["name"]].append(float(leg["timing_ms"]))
            if capture["fused"].get("timing_ms") is not None:
                fusion.append(float(capture["fused"]["timing_ms"]))
            if capture["reranked"].get("timing_ms") is not None:
                rerank.append(float(capture["reranked"]["timing_ms"]))

    def stats(values: list[float]) -> dict:
        return {"n": len(values), "p50": percentile(values, .5), "p95": percentile(values, .95)}

    return {"cases": record_count, "per_leg_ms": {name: stats(values) for name, values in sorted(legs.items())},
            "fusion_ms": stats(fusion), "rerank_ms": stats(rerank),
            "capture_wall_seconds_including_warmups_and_parity":
                json.loads((BASE / "manifest.json").read_text(encoding="utf-8"))["elapsed_seconds"]}


# The stage capture is deliberately streamed because it contains large candidate texts.


def main() -> None:
    key = judge_key(MODEL, think=True)
    if key.prompt_sha256 != PROMPT_SHA256:
        raise SystemExit(f"Unexpected judge prompt: {key.prompt_sha256}")
    output = {
        "evaluation": "development_only",
        "judge": {"model": MODEL, "prompt_sha256": key.prompt_sha256, "think": True,
                  "independent_human_calibration": False,
                  "calibration_skipped": True,
                  "calibration_reason": "No independent human calibration labels are available; all results are provisional."},
        "datasets": {name: run_dataset(name, key) for name in MODES_DATASETS},
        "judge_run_provenance": {name: write_judge_provenance(name, key) for name in MODES_DATASETS},
        "current_capture_stage_timings": stage_timings(),
        "artifacts_sha256": {
            name: sha256(BASE / name) for name in
            ("target.jsonl", "stages.jsonl", "per-case.jsonl", "flask-evidence-parity.json")
        },
    }
    path = BASE / "priority1-paired-analysis.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path), "datasets": list(output["datasets"]),
                      "judge": output["judge"], "capture_cases": output["current_capture_stage_timings"]["cases"]},
                     ensure_ascii=False, indent=2))


MODES_DATASETS = ("ip-200-fanout", "crossdomain-60-fanout")

if __name__ == "__main__":
    main()

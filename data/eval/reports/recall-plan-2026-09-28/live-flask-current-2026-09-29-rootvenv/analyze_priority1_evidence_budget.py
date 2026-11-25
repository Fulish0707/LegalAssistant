#!/usr/bin/env python3
"""Measure quote-survival ceilings under the current Flask evidence budget."""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE
while not (ROOT / "src" / "legal_rag").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from legal_rag.evaluation.judgment_store import JudgmentStore
from legal_rag.evaluation.recall_report import prepare_evidence
from legal_rag.evaluation.relevance_judge import judge_key, _normalize, _strip_quote_edges

MODEL = "ollama:qwen3:8b"
PROMPT_SHA256 = "7a2b826ac9cb93494241c98ccc829755d2af0ee37ebd8e3178cdbd798ecb596c"
CHAR_LIMIT = 1200
MAX_DOCUMENTS = 5
SEPARATOR = "\n…\n"
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。])\s+|(?<=…)\s+")
DATASETS = ("ip-200-fanout", "crossdomain-60-fanout")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def canonical_quote(text: str) -> str:
    return _strip_quote_edges(_normalize(text or ""))


def normalized_with_offsets(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Collapse whitespace and keep the corresponding source character ranges."""
    chars: list[str] = []
    offsets: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        if text[index].isspace():
            end = index + 1
            while end < len(text) and text[end].isspace():
                end += 1
            if chars:
                chars.append(" ")
                offsets.append((index, end))
            index = end
        else:
            chars.append(text[index])
            offsets.append((index, index + 1))
            index += 1
    if chars and chars[-1] == " ":
        chars.pop()
        offsets.pop()
    return "".join(chars), offsets


def quote_occurrences(text: str, quote: str) -> list[tuple[int, int]]:
    normalized, offsets = normalized_with_offsets(text)
    needle = canonical_quote(quote)
    if not needle:
        return []
    found = []
    start = 0
    while True:
        position = normalized.find(needle, start)
        if position < 0:
            break
        end_position = position + len(needle) - 1
        found.append((offsets[position][0], offsets[end_position][1]))
        start = position + 1
    return found


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    cursor = 0
    for match in SENTENCE_BOUNDARY.finditer(text):
        left, right = cursor, match.start()
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            spans.append((left, right))
        cursor = match.end()
    left, right = cursor, len(text)
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    if left < right:
        spans.append((left, right))
    return spans


def pair_set_for_window(occurrences: dict[tuple[str, str], list[tuple[int, int]]],
                        start: int, end: int) -> set[tuple[str, str]]:
    return {pair for pair, locations in occurrences.items()
            if any(start <= left and right <= end for left, right in locations)}


def best_window(text: str, occurrences: dict[tuple[str, str], list[tuple[int, int]]],
                *, sentence_bounded: bool) -> tuple[str, set[tuple[str, str]]]:
    """Choose a label-aware <=1200-char window, keeping prefix as a fallback."""
    if len(text) <= CHAR_LIMIT:
        return text, pair_set_for_window(occurrences, 0, len(text))

    candidates: list[tuple[int, int]] = [(0, CHAR_LIMIT)]
    if sentence_bounded:
        spans = sentence_spans(text)
        for first in range(len(spans)):
            for last in range(first, len(spans)):
                start, end = spans[first][0], spans[last][1]
                if end - start > CHAR_LIMIT:
                    break
                candidates.append((start, end))
    else:
        max_start = len(text) - CHAR_LIMIT
        starts = {0, max_start}
        for locations in occurrences.values():
            for left, right in locations:
                low = max(0, right - CHAR_LIMIT)
                high = min(left, max_start)
                if low <= high:
                    starts.add(low)
                    starts.add(high)
        candidates.extend((start, min(len(text), start + CHAR_LIMIT)) for start in sorted(starts))

    best_text = text[:CHAR_LIMIT]
    best_pairs = pair_set_for_window(occurrences, 0, len(best_text))
    for start, end in candidates[1:]:
        visible = pair_set_for_window(occurrences, start, end)
        if len(visible) > len(best_pairs):
            best_text, best_pairs = text[start:end], visible
    return best_text, best_pairs


def load_quotes(case_id: str, ranked: list[dict], pool: dict, judgments: dict) -> list[dict]:
    quotes = []
    for hit in ranked:
        pair = (case_id, hit["chunk_id"])
        judgment = judgments.get(pair)
        if not judgment or judgment.get("grade") not in (0, 1, 2):
            raise ValueError(f"Missing valid current top-10 judgment: {pair}")
        if judgment["grade"] != 2:
            continue
        details = judgment.get("details") or {}
        quote = details.get("evidence_quote") or ""
        if details.get("quote_verified") and quote:
            pool_row = pool[pair]
            quotes.append({"pair": pair, "document_id": hit["document_id"], "quote": quote,
                           "chunk_id": hit["chunk_id"], "chunk_text": pool_row["text"]})
    return quotes


def build_case_docs(case_id: str, ranked: list[dict], pool: dict) -> list[dict]:
    docs: dict[str, dict] = {}
    for position, hit in enumerate(ranked):
        chunk = pool[(case_id, hit["chunk_id"])]
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        document_id = hit.get("document_id") or ("hit", position)
        if document_id not in docs:
            docs[document_id] = {"document_id": document_id, "parts": [], "seen": set()}
        doc = docs[document_id]
        if text in doc["seen"]:
            continue
        doc["seen"].add(text)
        doc["parts"].append(text)
    result = []
    for rank, doc in enumerate(docs.values()):
        joined = SEPARATOR.join(doc["parts"])
        result.append({"document_id": doc["document_id"], "rank": rank, "joined": joined})
    return result


def make_case_views(case_id: str, ranked: list[dict], pool: dict, judgments: dict,
                    captured: dict) -> dict:
    quotes = load_quotes(case_id, ranked, pool, judgments)
    docs = build_case_docs(case_id, ranked, pool)

    # Verify the reassembled baseline against the already captured Flask evidence.
    replay = prepare_evidence([
        {"chunk_id": hit["chunk_id"], "document_id": hit["document_id"],
         "chunk": {"text": pool[(case_id, hit["chunk_id"])]["text"]}}
        for hit in ranked
    ])
    expected = [(row["document_id"], row["text"]) for row in captured["evidence"]]
    actual = [(row["document_id"], row["text"]) for row in replay]
    if actual != expected:
        raise ValueError(f"Flask formatter parity failed for {case_id}")

    quotes_by_doc: dict[str, list[dict]] = defaultdict(list)
    for quote in quotes:
        quotes_by_doc[quote["document_id"]].append(quote)

    visible_by_chunk = {}
    for document in replay:
        for segment in document["segments"]:
            key = (case_id, segment["chunk_id"])
            visible_by_chunk[key] = visible_by_chunk.get(key, "") + document["text"][
                segment["evidence_start"]:segment["evidence_end"]]
    pair_chunk_visible = {quote["pair"] for quote in quotes
                          if squash(quote["quote"]) in squash(visible_by_chunk.get(quote["pair"], ""))}
    pair_chunk_visible_canonical = {quote["pair"] for quote in quotes
                                    if canonical_quote(quote["quote"]) in _normalize(
                                        visible_by_chunk.get(quote["pair"], ""))}

    for doc in docs:
        occurrence_map = {}
        for quote in quotes_by_doc.get(doc["document_id"], []):
            occurrence_map[quote["pair"]] = quote_occurrences(doc["joined"], quote["quote"])
        doc["occurrences"] = occurrence_map
        doc["prefix"] = doc["joined"][:CHAR_LIMIT]
        doc["prefix_pairs"] = pair_set_for_window(occurrence_map, 0, len(doc["prefix"]))
        doc["sentence_excerpt"], doc["sentence_pairs"] = best_window(
            doc["joined"], occurrence_map, sentence_bounded=True)
        doc["char_excerpt"], doc["char_pairs"] = best_window(
            doc["joined"], occurrence_map, sentence_bounded=False)
    visible_quote_locations = sum(bool(locations) for doc in docs for locations in doc["occurrences"].values())
    return {"case_id": case_id, "quotes": quotes, "docs": docs,
            "pair_chunk_visible": pair_chunk_visible,
            "pair_chunk_visible_canonical": pair_chunk_visible_canonical,
            "unlocatable_quotes": len(quotes) - visible_quote_locations}


def select_docs(case: dict, view: str) -> tuple[list[dict], set[tuple[str, str]]]:
    docs = case["docs"]
    count = min(MAX_DOCUMENTS, len(docs))
    if count == 0:
        return [], set()
    best = None
    best_visible: set[tuple[str, str]] = set()
    for subset in itertools.combinations(docs, count):
        visible: set[tuple[str, str]] = set()
        for doc in subset:
            visible.update(doc[f"{view}_pairs"])
        if best is None or len(visible) > len(best_visible):
            best, best_visible = subset, visible
    return list(best or ()), best_visible


def variants_for_case(case: dict) -> dict[str, dict]:
    current_docs = case["docs"][:MAX_DOCUMENTS]
    return {
        "current_top5_prefix": {
            "docs": current_docs,
            "visible": set().union(*(doc["prefix_pairs"] for doc in current_docs)) if current_docs else set(),
        },
        "current_top5_sentence_window_oracle": {
            "docs": current_docs,
            "visible": set().union(*(doc["sentence_pairs"] for doc in current_docs)) if current_docs else set(),
        },
        "current_top5_char_window_oracle": {
            "docs": current_docs,
            "visible": set().union(*(doc["char_pairs"] for doc in current_docs)) if current_docs else set(),
        },
        "best5_prefix_oracle": dict(zip(("docs", "visible"), select_docs(case, "prefix"))),
        "best5_sentence_window_oracle": dict(zip(("docs", "visible"), select_docs(case, "sentence"))),
        "best5_char_window_oracle": dict(zip(("docs", "visible"), select_docs(case, "char"))),
    }


def summarize_dataset(name: str, cases: list[dict], variants: dict[str, dict[str, dict]]) -> dict:
    eligible = {case["case_id"]: {quote["pair"] for quote in case["quotes"]}
                for case in cases if case["quotes"]}
    result = {"case_count": len(cases), "cases_with_verified_grade2_quote": len(eligible),
              "verified_grade2_quotes": sum(len(pairs) for pairs in eligible.values()),
              "unlocatable_quotes": sum(case["unlocatable_quotes"] for case in cases),
              "variants": {}}
    variant_names = list(next(iter(variants.values()))) if variants else []
    base_visible = {case_id: rows["current_top5_prefix"] for case_id, rows in variants.items()}
    base_pairs = sum(len(view["visible"] & eligible[case_id]) for case_id, view in base_visible.items()
                     if case_id in eligible)
    for variant_name in variant_names:
        visible_pairs = 0
        visible_cases = 0
        new_cases = 0
        doc_counts = []
        selected_doc_sets = {}
        for case_id, pairs in eligible.items():
            view = variants[case_id][variant_name]
            visible = set(view["visible"]) & pairs
            visible_pairs += len(visible)
            visible_cases += int(bool(visible))
            new_cases += int(bool(visible) and not bool(base_visible[case_id]["visible"] & pairs))
            doc_counts.append(len(view["docs"]))
            selected_doc_sets[case_id] = [doc["document_id"] for doc in view["docs"]]
        result["variants"][variant_name] = {
            "visible_quote_pairs": visible_pairs,
            "verified_quote_pairs": result["verified_grade2_quotes"],
            "pair_retention": visible_pairs / result["verified_grade2_quotes"] if result["verified_grade2_quotes"] else None,
            "visible_cases_with_quote": visible_cases,
            "eligible_cases": len(eligible),
            "case_retention": visible_cases / len(eligible) if eligible else None,
            "visible_cases_among_all_capture_cases": sum(
                bool(variants.get(case["case_id"], {}).get(variant_name, {}).get("visible")) for case in cases
            ),
            "newly_visible_cases_vs_current": new_cases,
            "pair_change_vs_current": visible_pairs - base_pairs,
            "mean_documents_shown_on_eligible_cases": statistics.mean(doc_counts) if doc_counts else 0,
            "selected_document_ids": selected_doc_sets,
        }
    return result


def main() -> None:
    key = judge_key(MODEL, think=True)
    if key.prompt_sha256 != PROMPT_SHA256:
        raise SystemExit(f"Unexpected judge prompt hash: {key.prompt_sha256}")

    capture_rows = {row["case_id"]: row for row in read_jsonl(BASE / "per-case.jsonl")}
    all_cases: dict[str, dict] = {}
    input_hashes = {"per-case.jsonl": sha256(BASE / "per-case.jsonl"),
                    "priority1-paired-analysis.json": sha256(BASE / "priority1-paired-analysis.json")}
    for name in DATASETS:
        pool_dir = BASE / "grading" / name
        runs = read_jsonl(pool_dir / "runs" / "flask_current_k100.jsonl")
        pool_rows = read_jsonl(pool_dir / "pool.jsonl")
        pool = {(row["case_id"], row["chunk_id"]): row for row in pool_rows}
        judgments = JudgmentStore(pool_dir / "judgments.jsonl").rows(key)
        input_hashes[f"grading/{name}/pool.jsonl"] = sha256(pool_dir / "pool.jsonl")
        input_hashes[f"grading/{name}/runs/flask_current_k100.jsonl"] = sha256(
            pool_dir / "runs" / "flask_current_k100.jsonl")
        input_hashes[f"grading/{name}/judgments.jsonl"] = sha256(pool_dir / "judgments.jsonl")
        for run in runs:
            case_id = run["case_id"]
            if case_id not in capture_rows:
                raise ValueError(f"No captured evidence for {case_id}")
            case = make_case_views(case_id, run["ranked"][:10], pool, judgments, capture_rows[case_id])
            case["dataset"] = name
            all_cases[case_id] = case

    per_case_variants = {case_id: variants_for_case(case) for case_id, case in all_cases.items()}
    summaries = {}
    for name in DATASETS:
        cases = [case for case in all_cases.values() if case["dataset"] == name]
        variants = {case["case_id"]: per_case_variants[case["case_id"]] for case in cases}
        summaries[name] = summarize_dataset(name, cases, variants)

    prior = json.loads((BASE / "priority1-paired-analysis.json").read_text(encoding="utf-8"))
    expected_baselines = {
        name: prior["datasets"][name]["current_flask_evidence_survival"]["grade2_quotes_visible_own_excerpt"]
        for name in DATASETS
    }
    for name, expected in expected_baselines.items():
        cases = [case for case in all_cases.values() if case["dataset"] == name]
        pair_chunk_actual = sum(len(case["pair_chunk_visible"]) for case in cases)
        pair_chunk_canonical = sum(len(case["pair_chunk_visible_canonical"]) for case in cases)
        document_actual = summaries[name]["variants"]["current_top5_prefix"]["visible_quote_pairs"]
        if pair_chunk_actual != expected:
            raise ValueError(f"Own-chunk baseline mismatch for {name}: {pair_chunk_actual} != {expected}")
        summaries[name]["baseline_reconciliation"] = {
            "previous_priority1_own_chunk_visible_raw": expected,
            "recomputed_own_chunk_visible_raw": pair_chunk_actual,
            "own_chunk_visible_after_verifier_normalization": pair_chunk_canonical,
            "source_document_visible_after_verifier_normalization": document_actual,
            "additional_quotes_visible_in_sibling_chunk_of_same_document": document_actual - pair_chunk_canonical,
            "additional_quotes_recovered_by_verifier_edge_stripping": pair_chunk_canonical - pair_chunk_actual,
        }

    result = {
        "evaluation": "development_only_counterfactual",
        "question": "With current top-10 ranks fixed, how much verified grade-2 quote survival is limited by the five-document and 1,200-character evidence budget?",
        "retrieval_or_serving_changed": False,
        "judge": {"model": MODEL, "prompt_sha256": key.prompt_sha256, "think": True,
                  "independent_human_calibration": False,
                  "limitation": "Provisional Qwen-selected quotes, verified against retrieved chunks; not human-validated support spans."},
        "method": {
            "ranked_hits": "Captured current Flask top 10, unchanged.",
            "document_candidates": "Distinct nonempty documents in those top 10, preserving first-ranked order.",
            "budget": "At most five documents, at most 1,200 characters per document, and therefore no more than 6,000 characters total.",
            "prefix": "Current Flask behavior: concatenate distinct chunk texts in rank order with newline-ellipsis-newline separators, then take the first 1,200 characters.",
            "quote_match": "Use the relevance judge verifier normalization: collapse whitespace and strip surrounding quote marks and ellipses. Also record the prior stricter whitespace-only own-chunk count for reconciliation.",
            "sentence_window": "Best contiguous sentence-bounded excerpt per document, up to 1,200 characters; current prefix remains an available fallback.",
            "char_window": "Best arbitrary contiguous excerpt per document, up to 1,200 characters; current prefix remains an available fallback.",
            "oracle_warning": "Document subsets and excerpt locations are selected using the same provisional grade-2 quotes being measured. All oracle results are optimistic ceilings, not deployable selector performance.",
            "coverage_scope": "A quote counts only when visible in its source document's excerpt. Duplicate text in another selected document is not credited, consistent with the prior own-excerpt baseline.",
        },
        "variants": {
            "current_top5_prefix": "Current Flask baseline.",
            "current_top5_sentence_window_oracle": "Keep the current first five documents; allow one label-aware sentence window per document.",
            "current_top5_char_window_oracle": "Keep the current first five documents; allow one label-aware contiguous character window per document.",
            "best5_prefix_oracle": "Choose the subset of up to five top-10 documents that maximizes quote-pair survival using current prefix excerpts.",
            "best5_sentence_window_oracle": "Choose up to five top-10 documents and their label-aware sentence windows jointly.",
            "best5_char_window_oracle": "Choose up to five top-10 documents and their label-aware contiguous character windows jointly.",
        },
        "datasets": summaries,
        "input_sha256": input_hashes,
    }
    output_json = BASE / "priority1-evidence-budget-analysis.json"
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["analysis_script_sha256"] = sha256(Path(__file__))
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(result, BASE / "priority1-evidence-budget-report.md")
    compact = {
        name: {
            variant: {key: row[key] for key in (
                "visible_quote_pairs", "verified_quote_pairs", "visible_cases_with_quote",
                "eligible_cases", "newly_visible_cases_vs_current", "pair_change_vs_current")}
            for variant, row in dataset["variants"].items()
        }
        for name, dataset in summaries.items()
    }
    print(json.dumps({"json": str(output_json), "report": str(BASE / "priority1-evidence-budget-report.md"),
                      "datasets": compact}, ensure_ascii=False, indent=2))


def write_report(data: dict, path: Path) -> None:
    labels = {
        "current_top5_prefix": "Current top 5 + prefix",
        "current_top5_sentence_window_oracle": "Same top 5 + sentence-window oracle",
        "current_top5_char_window_oracle": "Same top 5 + character-window oracle",
        "best5_prefix_oracle": "Best 5 + prefix oracle",
        "best5_sentence_window_oracle": "Best 5 + sentence-window oracle",
        "best5_char_window_oracle": "Best 5 + character-window oracle",
    }
    lines = [
        "# Priority 1 follow-up — Evidence-budget counterfactual",
        "",
        "- **Date:** 2026-09-29",
        "- **Result type:** Development-only offline counterfactual; retrieval and serving unchanged",
        "- **Scope:** Current captured top 10 for the existing 200 IP and 60 mixed-domain questions",
        "",
        "## Method",
        "",
        "Each grade-2 top-10 pair contributes its locally judged, quote-verified Qwen quote. The relevance judge verifier normalization is used: whitespace is collapsed and surrounding quote marks and ellipses are stripped. The stricter prior own-chunk count is also reconciled below. The analysis replays Flask's current formatter and confirms its reconstructed ordered evidence equals the captured evidence for every case. It then compares the baseline with alternatives under the same cap of five documents and 1,200 characters per document (at most 6,000 characters per question).",
        "",
        "The document-choice and excerpt-placement variants are **label-aware upper bounds**: they use the same provisional quotes they score to pick documents or excerpt locations. They show where the current budget can lose those quotes; they do not estimate how a real selector would perform. The sentence-window variant keeps the current prefix as a fallback. The character-window variant allows any contiguous excerpt of the same length.",
        "",
        "## Baseline reconciliation",
        "",
    ]
    for dataset, summary in data["datasets"].items():
        rec = summary["baseline_reconciliation"]
        short = "IP" if dataset.startswith("ip-") else "Mixed domains"
        lines.append(f"- {short}: the prior whitespace-only own-chunk count reproduces at {rec['recomputed_own_chunk_visible_raw']}/{rec['previous_priority1_own_chunk_visible_raw']}. Applying the judge verifier's edge stripping adds {rec['additional_quotes_recovered_by_verifier_edge_stripping']}; counting matching quote text in a sibling chunk of the same displayed document adds {rec['additional_quotes_visible_in_sibling_chunk_of_same_document']} more, for {rec['source_document_visible_after_verifier_normalization']} source-document quotes.")
    lines.extend([
        "",
        "## Quote survival",
        "",
        "Counts are shown as visible source-document quotes / all verified grade-2 quotes. Case counts use questions with at least one verified grade-2 quote in the top 10 as the denominator. Cases newly visible are the number with no baseline quote that gain at least one under the variant.",
        "",
        "| Dataset | Evidence view | Quote pairs | Pair retention | Cases with a visible quote | Case retention | New cases vs baseline | Pair change vs baseline |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    order = list(labels)
    for dataset, summary in data["datasets"].items():
        short = "IP (200)" if dataset.startswith("ip-") else "Mixed domains (60)"
        for variant in order:
            item = summary["variants"][variant]
            lines.append(
                f"| {short} | {labels[variant]} | {item['visible_quote_pairs']}/{item['verified_quote_pairs']} "
                f"| {item['pair_retention']:.1%} | {item['visible_cases_with_quote']}/{item['eligible_cases']} "
                f"| {item['case_retention']:.1%} | {item['newly_visible_cases_vs_current']} "
                f"| {item['pair_change_vs_current']:+d} |"
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- The same-top-five rows isolate excerpt placement. Compare their case gains as well as pair gains; a high pair-count gain can come from many quotes on questions that already had some visible support.",
        "- The best-five rows isolate document selection when using prefixes, then show the combined document-plus-excerpt ceiling. These are hindsight oracles, not candidate production behavior.",
        "- The gap between sentence-window and character-window ceilings indicates how much of the possible gain depends on breaking sentence boundaries or selecting inside a long sentence.",
        "- These are quote-retention measurements, not answer-quality measurements or classic Recall@k. Human support-span labels and independent calibration remain unavailable.",
        "",
        "## Provenance",
        "",
        f"Judge: `{data['judge']['model']}`, prompt `{data['judge']['prompt_sha256']}`, `think=true`. "
        "Independent human calibration was unavailable. Input hashes and the reproducible analysis script are recorded in `priority1-evidence-budget-analysis.json`.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

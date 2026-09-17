"""Graded retrieval metrics over pooled LLM judgments.

Gains are ``2**grade - 1``. Unjudged chunks count as grade 0 in the metrics, and the fraction
judged is reported as ``judged_coverage@10``. Chunks whose judgment failed to parse count as
judged (so a rerun cannot be blocked by them), take gain 0, are left out of the ideal ranking, and
are reported as ``failed@10`` per case and ``failed_top10_total`` in the summary. A comparison is valid only when every run's
chunk-level coverage is 1.0. The ideal ranking uses every judged chunk for the case, so it does
not depend on the run being scored.

Document level: a document takes the rank of its first retrieved chunk and the maximum grade of
its judged chunks (including chunks judged only for other runs). A document counts as judged
when any of its chunks is graded or failed, even if the chunk this run retrieved for it was not, so
document-level coverage can read 1.0 while a retrieved chunk was never graded. Chunk-level
coverage is therefore the strict validity check.

A case whose judged chunks are all grade 0 has an ideal DCG of 0: its nDCG is undefined
(``None``), it is excluded from the nDCG mean and counted in ``cases_without_relevant``, but it
still counts as 0 for success, MRR and precision. MRR (``mrr@<depth>``) and success@k are limited to
the pool depth, because deeper ranks were never pooled and are mostly unjudged; success@20 is
dropped when the pool is shallower than 20. Coverage is reported at 10 and at the largest success
cut-off (``judged_coverage@20``), and validity requires chunk coverage 1.0 at both.
Pooled recall is deliberately omitted.

Relevance threshold: ``success@k``, ``mrr@<depth>``, ``precision@10`` and ``ndcg@10_binary`` count a
chunk (or document) as relevant when its grade is at least ``threshold`` (1 or 2; default 2).
``ndcg@10_binary`` uses gain 1 for relevant items and an ideal ranking of every judged item at or
above the threshold; it is ``None`` when there is none (counted in ``cases_without_relevant_binary``).
Graded ``ndcg@10`` and ``precision@10_grade1``/``precision@10_grade2`` do not depend on the threshold.
"""
from __future__ import annotations

import math
from statistics import mean

SUCCESS_KS = (5, 10, 20)
LEVELS = ('chunk', 'document')
POOL_DEPTH = 20
THRESHOLDS = (1, 2)


def success_ks(depth: int) -> list[int]:
    """Success cut-offs within the judged (pooled) depth; deeper ones would score unjudged chunks."""
    return [k for k in SUCCESS_KS if k <= depth]


def coverage_ks(depth: int) -> list[int]:
    """Coverage is reported at 10 and at the largest success cut-off, never deeper than the pool."""
    return sorted({min(10, depth), *success_ks(depth)[-1:]})


def _dcg(gains: list[int]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(gains, 1))


def case_metrics(ranked: list[dict], grades: dict[str, int], *, level: str,
                 documents: dict[str, str] | None = None, failed: set[str] | frozenset = frozenset(),
                 depth: int = POOL_DEPTH, threshold: int = 2) -> dict:
    """``grades`` maps chunk id to grade for this case; ``documents`` maps judged chunk id to
    document id (from the store, so chunks judged for other runs still count in the ideal).
    ``failed`` holds chunk ids whose latest judgment is a parse failure: they count as judged for
    coverage, take gain 0, stay out of the ideal ranking and are counted in ``failed@10``. ``depth`` is the pool depth: MRR and success stop there.
    ``threshold`` is the minimum grade counted as relevant by the headline metrics."""
    if threshold not in THRESHOLDS:
        raise ValueError(f'threshold must be 1 or 2, got {threshold!r}')
    failed = set(failed) - set(grades)
    if level == 'document':
        documents = {**{h['chunk_id']: h['document_id'] for h in ranked}, **(documents or {})}
        labels: dict[str, int] = {}
        for chunk_id, grade in grades.items():
            doc = documents[chunk_id]
            labels[doc] = max(labels.get(doc, 0), grade)
        failed_items = {documents[chunk_id] for chunk_id in failed} - set(labels)
        items = list(dict.fromkeys(hit['document_id'] for hit in ranked))
    elif level == 'chunk':
        items, labels, failed_items = [h['chunk_id'] for h in ranked], dict(grades), failed
    else:
        raise ValueError(f'unknown level: {level!r}')
    gains = [labels.get(item, 0) for item in items]
    ideal = _dcg(sorted(labels.values(), reverse=True)[:10])
    relevant = [int(g >= threshold) for g in gains]
    ideal_binary = _dcg(sorted((int(g >= threshold) for g in labels.values()), reverse=True)[:10])
    first = next((i for i, r in enumerate(relevant[:depth], 1) if r), None)
    top10 = items[:10]

    def coverage(k: int) -> float:
        top = items[:k]
        return sum(item in labels or item in failed_items for item in top) / len(top) if top else 1.0
    return {
        'ndcg@10': _dcg(gains[:10]) / ideal if ideal else None,
        'ndcg@10_binary': _dcg(relevant[:10]) / ideal_binary if ideal_binary else None,
        **{f'success@{k}': int(any(relevant[:k])) for k in success_ks(depth)},
        f'mrr@{depth}': 1 / first if first else 0.0,
        'precision@10': sum(relevant[:10]) / 10,
        'precision@10_grade1': sum(g >= 1 for g in gains[:10]) / 10,
        'precision@10_grade2': sum(g == 2 for g in gains[:10]) / 10,
        **{f'judged_coverage@{k}': coverage(k) for k in coverage_ks(depth)},
        'failed@10': sum(item in failed_items for item in top10),
    }


def per_case_metrics(runs: dict[str, list[dict]], grades: dict[tuple[str, str], int],
                     documents: dict[tuple[str, str], str],
                     failures: set[tuple[str, str]] | frozenset = frozenset(), depth: int = POOL_DEPTH,
                     threshold: int = 2) -> dict:
    """Return ``{mode: {level: [{'case_id': ..., **metrics}]}}``.

    Raises ``ValueError`` if a mode has no cases or modes cover different case ids."""
    case_sets = {}
    for mode, rows in runs.items():
        if not rows:
            raise ValueError(f'mode {mode!r} has no cases')
        case_sets[mode] = frozenset(row['case_id'] for row in rows)
    if len(set(case_sets.values())) > 1:
        raise ValueError(f'modes must cover the same cases for a paired comparison: '
                         f'{ {m: len(c) for m, c in case_sets.items()} }')
    per_case_grades: dict[str, dict[str, int]] = {}
    per_case_docs: dict[str, dict[str, str]] = {}
    for (case_id, chunk_id), grade in grades.items():
        per_case_grades.setdefault(case_id, {})[chunk_id] = grade
        per_case_docs.setdefault(case_id, {})[chunk_id] = documents[(case_id, chunk_id)]
    per_case_failed: dict[str, set[str]] = {}
    for case_id, chunk_id in failures:
        per_case_failed.setdefault(case_id, set()).add(chunk_id)
        if (case_id, chunk_id) in documents:
            per_case_docs.setdefault(case_id, {})[chunk_id] = documents[(case_id, chunk_id)]
    return {
        mode: {
            level: [{'case_id': row['case_id'],
                     **case_metrics(row['ranked'], per_case_grades.get(row['case_id'], {}), level=level,
                                    documents=per_case_docs.get(row['case_id'], {}),
                                    failed=per_case_failed.get(row['case_id'], set()), depth=depth,
                                    threshold=threshold)}
                    for row in rows]
            for level in LEVELS
        }
        for mode, rows in runs.items()
    }


def summarize_runs(runs: dict[str, list[dict]], grades: dict[tuple[str, str], int],
                   documents: dict[tuple[str, str], str],
                   failures: set[tuple[str, str]] | frozenset = frozenset(), depth: int = POOL_DEPTH,
                   threshold: int = 2) -> dict:
    """``failures`` is ``set(store.failures(key))`` and ``depth`` is pool.json ``pool_depth``;
    see ``case_metrics``."""
    summary = {}
    for mode, levels in per_case_metrics(runs, grades, documents, failures, depth, threshold).items():
        summary[mode] = {}
        for level, results in levels.items():
            ndcgs = [r['ndcg@10'] for r in results if r['ndcg@10'] is not None]
            binary = [r['ndcg@10_binary'] for r in results if r['ndcg@10_binary'] is not None]
            keys = [k for k in results[0] if k not in ('case_id', 'ndcg@10', 'ndcg@10_binary', 'failed@10')
                    and not k.startswith('judged_coverage@')]
            summary[mode][level] = {
                'cases': len(results), 'cases_without_relevant': len(results) - len(ndcgs),
                'cases_without_relevant_binary': len(results) - len(binary),
                'ndcg@10': mean(ndcgs) if ndcgs else None,
                'ndcg@10_binary': mean(binary) if binary else None,
                **{k: mean(r[k] for r in results) for k in keys},
                **{f'{stat.__name__}_judged_coverage@{k}': stat(r[f'judged_coverage@{k}'] for r in results)
                   for k in coverage_ks(depth) for stat in (mean, min)},
                'failed_top10_total': sum(r['failed@10'] for r in results),
            }
    return summary

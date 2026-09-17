"""Graded retrieval metrics over pooled LLM judgments.

Gains are ``2**grade - 1``. Unjudged chunks count as grade 0 in the metrics, and the fraction
judged is reported as ``judged_coverage@10``. A comparison is valid only when every run's
chunk-level coverage is 1.0. The ideal ranking uses every judged chunk for the case, so it does
not depend on the run being scored.

Document level: a document takes the rank of its first retrieved chunk and the maximum grade of
its judged chunks (including chunks judged only for other runs). A document counts as judged
when any of its chunks is judged, even if the chunk this run retrieved for it was not, so
document-level coverage can read 1.0 while a retrieved chunk was never graded. Chunk-level
coverage is therefore the strict validity check.

A case whose judged chunks are all grade 0 has an ideal DCG of 0: its nDCG is undefined
(``None``), it is excluded from the nDCG mean and counted in ``cases_without_relevant``, but it
still counts as 0 for success, MRR and precision. MRR is limited to the recorded ranking depth.
Pooled recall is deliberately omitted.
"""
from __future__ import annotations

import math
from statistics import mean

SUCCESS_KS = (5, 10, 20)
LEVELS = ('chunk', 'document')


def _dcg(gains: list[int]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(gains, 1))


def case_metrics(ranked: list[dict], grades: dict[str, int], *, level: str,
                 documents: dict[str, str] | None = None) -> dict:
    """``grades`` maps chunk id to grade for this case; ``documents`` maps judged chunk id to
    document id (from the store, so chunks judged for other runs still count in the ideal)."""
    if level == 'document':
        documents = {**{h['chunk_id']: h['document_id'] for h in ranked}, **(documents or {})}
        labels: dict[str, int] = {}
        for chunk_id, grade in grades.items():
            doc = documents[chunk_id]
            labels[doc] = max(labels.get(doc, 0), grade)
        items = list(dict.fromkeys(hit['document_id'] for hit in ranked))
    elif level == 'chunk':
        items, labels = [h['chunk_id'] for h in ranked], dict(grades)
    else:
        raise ValueError(f'unknown level: {level!r}')
    gains = [labels.get(item, 0) for item in items]
    ideal = _dcg(sorted(labels.values(), reverse=True)[:10])
    first = next((i for i, g in enumerate(gains, 1) if g == 2), None)
    top10 = items[:10]
    return {
        'ndcg@10': _dcg(gains[:10]) / ideal if ideal else None,
        **{f'success@{k}': int(any(g == 2 for g in gains[:k])) for k in SUCCESS_KS},
        'mrr': 1 / first if first else 0.0,
        'precision@10_grade1': sum(g >= 1 for g in gains[:10]) / 10,
        'precision@10_grade2': sum(g == 2 for g in gains[:10]) / 10,
        'judged_coverage@10': sum(item in labels for item in top10) / len(top10) if top10 else 1.0,
    }


def per_case_metrics(runs: dict[str, list[dict]], grades: dict[tuple[str, str], int],
                     documents: dict[tuple[str, str], str]) -> dict:
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
    return {
        mode: {
            level: [{'case_id': row['case_id'],
                     **case_metrics(row['ranked'], per_case_grades.get(row['case_id'], {}), level=level,
                                    documents=per_case_docs.get(row['case_id'], {}))}
                    for row in rows]
            for level in LEVELS
        }
        for mode, rows in runs.items()
    }


def summarize_runs(runs: dict[str, list[dict]], grades: dict[tuple[str, str], int],
                   documents: dict[tuple[str, str], str]) -> dict:
    summary = {}
    for mode, levels in per_case_metrics(runs, grades, documents).items():
        summary[mode] = {}
        for level, results in levels.items():
            ndcgs = [r['ndcg@10'] for r in results if r['ndcg@10'] is not None]
            keys = [k for k in results[0] if k not in ('case_id', 'ndcg@10', 'judged_coverage@10')]
            summary[mode][level] = {
                'cases': len(results), 'cases_without_relevant': len(results) - len(ndcgs),
                'ndcg@10': mean(ndcgs) if ndcgs else None,
                **{k: mean(r[k] for r in results) for k in keys},
                'mean_judged_coverage@10': mean(r['judged_coverage@10'] for r in results),
                'min_judged_coverage@10': min(r['judged_coverage@10'] for r in results),
            }
    return summary

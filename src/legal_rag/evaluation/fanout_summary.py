"""Judgment-free comparison of recorded runs: gold documents found, and which corpus answered.

A fan-out run surfaces chunks nobody has judged, so its graded metrics are depressed by
construction until judging catches up. These two summaries need no judge at all: the gold
document of each case is known from the target, and a hit's corpus is known from the processed
corpora. They answer whether fanning out loses the gold document, and whether an unfiltered
fan-out still lets the right corpus answer.
"""
from __future__ import annotations

from .retrieval_metrics import mrr

SUCCESS_DEPTHS = (1, 5, 10)


def _ranked_documents(row: dict, depth: int) -> list[str]:
    return [hit['document_id'] for hit in row['ranked'][:depth]]


def gold_document_metrics(runs: dict[str, list[dict]], gold: dict[str, set[str]], *,
                          depth: int) -> dict[str, dict[str, float]]:
    """Success@k and MRR over gold documents, per run, at ``depth``.

    ``gold`` maps a case id to its relevant document ids; a run row for an unknown case raises,
    since a metric over a case with no gold would silently count as a miss.
    """
    summary: dict[str, dict[str, float]] = {}
    for name, rows in runs.items():
        scores: dict[str, float] = {f'success@{k}': 0.0 for k in SUCCESS_DEPTHS if k <= depth}
        scores[f'mrr@{depth}'] = 0.0
        for row in rows:
            if row['case_id'] not in gold:
                raise ValueError(f'{name}: {row["case_id"]} has no gold documents in the target')
            ranked = _ranked_documents(row, depth)
            relevant = gold[row['case_id']]
            for k in SUCCESS_DEPTHS:
                if k <= depth:
                    scores[f'success@{k}'] += float(bool(set(ranked[:k]) & relevant))
            scores[f'mrr@{depth}'] += mrr(ranked, relevant)
        cases = max(len(rows), 1)
        summary[name] = {'cases': len(rows), **{key: value / cases for key, value in scores.items()}}
    return summary


def corpus_mix(runs: dict[str, list[dict]], corpora: dict[str, set[str]], *,
               depth: int) -> dict[str, dict[str, float]]:
    """Share of each run's top-``depth`` slots whose document belongs to each corpus.

    ``corpora`` maps a corpus version to its document ids. A document in no corpus is counted
    under 'unknown' rather than dropped, so the shares always sum to 1.
    """
    mix: dict[str, dict[str, float]] = {}
    for name, rows in runs.items():
        counts = {version: 0 for version in corpora}
        counts['unknown'] = 0
        slots = 0
        for row in rows:
            for document_id in _ranked_documents(row, depth):
                slots += 1
                owner = next((version for version, ids in corpora.items() if document_id in ids), 'unknown')
                counts[owner] += 1
        mix[name] = {'slots': slots, **{version: count / max(slots, 1) for version, count in counts.items()}}
    return mix

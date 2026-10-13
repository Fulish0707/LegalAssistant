"""Re-fuse recorded leg runs under different RRF weights, without retrieving again.

A pool directory records one run file per mode. When the evaluation applies a filter that
leaves a single corpus answering, ``runs/dense.jsonl`` and ``runs/sparse.jsonl`` are exactly
the ranked lists ``RRFFuser`` was given, so a differently weighted hybrid can be recomputed
from them. Under a multi-corpus fan-out they are already fused across corpora and this
shortcut is invalid.

:func:`refuse` mirrors ``legal_rag.retrieval.fusion.RRFFuser.fuse`` step for step: weights
default to 1.0 per leg, scores are ``weight / (rrf_k + rank)`` summed over the legs that saw
the chunk, ties break on ascending chunk id, the list is truncated to the depth and only then
are non-positive scores dropped (so a chunk seen by a zero-weighted leg alone disappears).
"""
from __future__ import annotations


def refuse(leg_runs: dict[str, list[dict]], weights: dict[str, float], *, rrf_k: int,
           depth: int) -> list[dict]:
    """Fuse the recorded leg runs under ``weights``, returning rows shaped like ``runs/*.jsonl``.

    ``leg_runs`` maps a leg name to its rows (``{'case_id': ..., 'ranked': [{'chunk_id',
    'document_id'}, ...]}``); ``weights`` is keyed by the same names. Every leg must cover the
    same cases in the same order, which is the order of the result.
    """
    if not leg_runs:
        raise ValueError('no leg runs to fuse')
    if depth < 1:
        raise ValueError('depth must be >= 1')
    orders = {leg: tuple(row['case_id'] for row in rows) for leg, rows in leg_runs.items()}
    if len(set(orders.values())) > 1:
        raise ValueError(f'legs must cover the same cases in the same order: '
                         f'{ {leg: len(order) for leg, order in orders.items()} }')
    order = next(iter(orders.values()))
    if len(set(order)) != len(order):
        raise ValueError('a leg run repeats a case_id')
    by_leg = {leg: {row['case_id']: row['ranked'] for row in rows} for leg, rows in leg_runs.items()}
    fused = []
    for case_id in order:
        scores: dict[str, float] = {}
        documents: dict[str, str] = {}
        for leg, ranked in by_leg.items():
            weight = weights.get(leg, 1.0)
            for rank, hit in enumerate(ranked[case_id], start=1):
                chunk_id = hit['chunk_id']
                scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (rrf_k + rank)
                documents.setdefault(chunk_id, hit['document_id'])
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:depth]
        fused.append({'case_id': case_id, 'ranked': [
            {'chunk_id': chunk_id, 'document_id': documents[chunk_id]}
            for chunk_id, score in ordered if score > 0]})
    return fused


def pair_ranks(runs: dict[str, list[dict]], depth: int) -> dict[tuple[str, str], dict[str, int]]:
    """``{(case_id, chunk_id): {mode: rank}}`` for every pair a run ranks within ``depth``."""
    if depth < 1:
        raise ValueError('depth must be >= 1')
    ranks: dict[tuple[str, str], dict[str, int]] = {}
    for mode, rows in runs.items():
        for row in rows:
            for rank, hit in enumerate(row['ranked'][:depth], start=1):
                ranks.setdefault((row['case_id'], hit['chunk_id']), {})[mode] = rank
    return ranks


def pending_pairs(runs: dict[str, list[dict]], judged_pairs: set[tuple[str, str]] | frozenset,
                  depth: int) -> list[tuple[str, str]]:
    """Distinct (case_id, chunk_id) pairs the runs pool within ``depth`` that are not yet judged."""
    return sorted(set(pair_ranks(runs, depth)) - set(judged_pairs))

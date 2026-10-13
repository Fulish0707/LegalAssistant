"""Rerank a pool's recorded fused run, without retrieving again.

A run file records the ranked chunk ids a mode returned. Reranking is a function of the query
and the candidates' text alone, so feeding a recorded fused run's chunks to a reranker
reproduces exactly what the live pipeline would return for that fused list — and the comparison
stays paired, since the baseline runs were measured on the same lists.
"""
from __future__ import annotations

from typing import Callable

from legal_rag.reranking import Reranker
from legal_rag.schemas import Chunk, ScoredChunk


def rerank_run(rows: list[dict], chunks: dict[str, Chunk], reranker: Reranker, questions: dict[str, str],
               *, depth: int, progress: Callable[[str], None] | None = None) -> list[dict]:
    """Rows shaped like ``runs/*.jsonl``, each case's candidates re-sorted by ``reranker``.

    ``chunks`` must hold every chunk the rows rank and ``questions`` every case; a gap raises
    rather than silently shortening a ranked list.
    """
    if depth < 1:
        raise ValueError('depth must be >= 1')
    reranked = []
    for index, row in enumerate(rows, 1):
        case_id = row['case_id']
        question = questions.get(case_id)
        if question is None:
            raise ValueError(f'no question for {case_id}; the runs and the pool disagree')
        candidates = []
        for rank, hit in enumerate(row['ranked'][:depth], 1):
            chunk = chunks.get(hit['chunk_id'])
            if chunk is None:
                raise ValueError(f'{hit["chunk_id"]} (case {case_id}) is not in the processed corpus')
            candidates.append(ScoredChunk(chunk=chunk, score=1.0 / rank, retriever='fused'))
        hits = reranker.rerank(question, candidates, depth)
        reranked.append({'case_id': case_id, 'ranked': [
            {'chunk_id': hit.chunk.id, 'document_id': hit.chunk.document_id} for hit in hits]})
        if progress and (index % 10 == 0 or index == len(rows)):
            progress(f'Reranked {index}/{len(rows)} cases')
    return reranked

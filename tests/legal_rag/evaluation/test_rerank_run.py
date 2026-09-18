from __future__ import annotations

import pytest

from legal_rag.evaluation.rerank_run import rerank_run
from legal_rag.schemas import Chunk, ChunkMetadata, DocumentType


def _chunks():
    return [Chunk(id=f'd{i}#p', document_id=f'd{i}', document_type=DocumentType.statute, path='p', order=0,
                  text=f'text {i}', title='t', metadata=ChunkMetadata(dataset='f', corpus_version='v'))
            for i in (1, 2, 3)]


class _Reranker:
    """Sorts by a per-chunk score, the way a cross-encoder would."""

    name = 'fake'

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.queries: list[str] = []

    def rerank(self, query, candidates, k):
        self.queries.append(query)
        ordered = sorted(candidates, key=lambda hit: -self.scores.get(hit.chunk.id, 0.0))
        return [hit.model_copy(update={'retriever': 'reranked'}) for hit in ordered[:k]]


def _fixtures():
    chunks = {chunk.id: chunk for chunk in _chunks()}
    rows = [{'case_id': 'c1', 'ranked': [{'chunk_id': chunk_id, 'document_id': chunk.document_id}
                                         for chunk_id, chunk in chunks.items()]}]
    return chunks, rows


def test_reorders_each_case_by_the_reranker_and_keeps_the_row_shape():
    chunks, rows = _fixtures()
    ids = list(chunks)
    reranker = _Reranker({ids[2]: 3.0, ids[0]: 2.0, ids[1]: 1.0})
    out = rerank_run(rows, chunks, reranker, {'c1': 'question?'}, depth=3)
    assert out == [{'case_id': 'c1', 'ranked': [{'chunk_id': ids[2], 'document_id': chunks[ids[2]].document_id},
                                                {'chunk_id': ids[0], 'document_id': chunks[ids[0]].document_id},
                                                {'chunk_id': ids[1], 'document_id': chunks[ids[1]].document_id}]}]
    assert reranker.queries == ['question?']


def test_truncates_to_depth():
    chunks, rows = _fixtures()
    out = rerank_run(rows, chunks, _Reranker({}), {'c1': 'q'}, depth=1)
    assert len(out[0]['ranked']) == 1


def test_a_missing_chunk_or_question_raises():
    chunks, rows = _fixtures()
    with pytest.raises(ValueError, match='not in the processed corpus'):
        rerank_run(rows, {}, _Reranker({}), {'c1': 'q'}, depth=3)
    with pytest.raises(ValueError, match='no question'):
        rerank_run(rows, chunks, _Reranker({}), {}, depth=3)


def test_reports_progress_per_case():
    chunks, rows = _fixtures()
    seen: list[str] = []
    rerank_run(rows, chunks, _Reranker({}), {'c1': 'q'}, depth=3, progress=seen.append)
    assert seen and '1/1' in seen[-1]

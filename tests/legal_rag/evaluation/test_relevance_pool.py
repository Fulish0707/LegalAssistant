import json

import pytest

from legal_rag.config import Settings
from legal_rag.evaluation.diagnostic_runner import deepen_retrieval
from legal_rag.evaluation.relevance_pool import build_pool, read_jsonl, write_pool
from legal_rag.schemas import Chunk, ChunkMetadata, DocumentType, ScoredChunk


def make_chunk(chunk_id, document_id=None, text='본문'):
    return Chunk(id=chunk_id, document_id=document_id or chunk_id.split('#')[0], document_type=DocumentType.judgment,
                 path='p', order=0, text=text, title='제목',
                 metadata=ChunkMetadata(dataset='d', corpus_version='ip-v1', authority='특허법원'))


class ListRetriever:
    def __init__(self, ranking):
        self.ranking, self.calls = ranking, []

    def retrieve(self, query, k, filters=None):
        self.calls.append((query, k, filters))
        return [ScoredChunk(chunk=make_chunk(c), score=1.0, retriever='fused') for c in self.ranking[:k]]


CASES = [dict(case_id='c1', question='질문', answer='답변', document_id='gold', legal_domain='intellectual_property')]


def test_pool_is_union_of_top_depth_plus_gold_with_each_modes_rank():
    retrievers = {'dense': ListRetriever(['a#0', 'b#0', 'c#0']), 'sparse': ListRetriever(['b#0', 'd#0'])}
    pool, runs = build_pool(CASES, retrievers, gold_chunks={'gold': [make_chunk('gold#0')]},
                            pool_depth=2, run_depth=3)
    rows = {row['chunk_id']: row for row in pool}
    assert set(rows) == {'a#0', 'b#0', 'd#0', 'gold#0'}           # c#0 is dense rank 3 > pool_depth
    assert rows['b#0']['ranks'] == {'dense': 2, 'sparse': 1}
    assert rows['gold#0']['ranks'] == {} and rows['gold#0']['is_gold_chunk']
    assert rows['a#0']['question'] == '질문' and rows['a#0']['reference_answer'] == '답변'
    assert rows['a#0']['search_query'] == '특허법원 a'
    assert runs['dense'] == [{'case_id': 'c1', 'ranked': [
        {'chunk_id': 'a#0', 'document_id': 'a'}, {'chunk_id': 'b#0', 'document_id': 'b'},
        {'chunk_id': 'c#0', 'document_id': 'c'}]}]


def test_retrieved_gold_chunk_keeps_its_ranks():
    pool, _ = build_pool(CASES, {'dense': ListRetriever(['x#0', 'gold#0'])},
                         gold_chunks={'gold': [make_chunk('gold#0'), make_chunk('gold#1', 'gold')]},
                         pool_depth=2, run_depth=2)
    rows = {row['chunk_id']: row for row in pool}
    assert rows['gold#0']['ranks'] == {'dense': 2} and rows['gold#0']['is_gold_chunk']
    assert rows['gold#1']['ranks'] == {} and rows['gold#1']['is_gold_chunk']
    assert not rows['x#0']['is_gold_chunk']
    assert len(pool) == 3


def test_retrieval_uses_run_depth_and_the_domain_filter():
    dense = ListRetriever(['a#0'])
    build_pool(CASES, {'dense': dense}, gold_chunks={}, pool_depth=1, run_depth=100)
    query, k, filters = dense.calls[0]
    assert (query, k, filters.legal_domain) == ('질문', 100, 'intellectual_property')


def test_pool_depth_cannot_exceed_run_depth():
    with pytest.raises(ValueError):
        build_pool(CASES, {'dense': ListRetriever([])}, gold_chunks={}, pool_depth=20, run_depth=10)


def test_progress_reports_every_ten_cases():
    cases = [dict(CASES[0], case_id=f'c{i}') for i in range(25)]
    messages = []
    build_pool(cases, {'dense': ListRetriever(['a#0'])}, gold_chunks={}, pool_depth=1, run_depth=1,
               progress=messages.append)
    assert len(messages) == 3


def test_write_refuses_existing_directory(tmp_path):
    pool, runs = build_pool(CASES, {'dense': ListRetriever(['a#0'])}, gold_chunks={}, pool_depth=1, run_depth=1)
    write_pool(tmp_path / 'out', pool, runs, {'target_sha256': 'x'})
    assert read_jsonl(tmp_path / 'out' / 'pool.jsonl')[0]['chunk_id'] == 'a#0'
    assert read_jsonl(tmp_path / 'out' / 'runs' / 'dense.jsonl') == runs['dense']
    assert json.loads((tmp_path / 'out' / 'pool.json').read_text())['pairs'] == 1
    with pytest.raises(FileExistsError):
        write_pool(tmp_path / 'out', pool, runs, {})


def test_deepen_retrieval_raises_depths_on_a_copy_only():
    settings = Settings(corpus={'version': 'v1'}, retrieval={'k_dense': 150, 'k_sparse': 20, 'k_fused': 30,
                                                             'rrf_k': 7, 'corpora': ['v1', 'v2']})
    deep = deepen_retrieval(settings, 100)
    assert (deep.retrieval.k_dense, deep.retrieval.k_sparse, deep.retrieval.k_fused) == (150, 100, 100)
    assert deep.retrieval.rrf_k == 7 and deep.retrieval.corpora == ['v1', 'v2']
    assert deep.retrieval.k_final == settings.retrieval.k_final
    assert (settings.retrieval.k_sparse, settings.retrieval.k_fused) == (20, 30)

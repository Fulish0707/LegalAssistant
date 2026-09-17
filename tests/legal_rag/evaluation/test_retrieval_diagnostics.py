import json
from datetime import date

import pytest

from legal_rag.evaluation.diagnostics import build_duplicate_question_index, diagnose_case, normalize_question
from legal_rag.schemas import Chunk, ChunkMetadata, ScoredChunk


def chunk(document_id, index=0):
    return Chunk(id=f'{document_id}#{index}', document_id=document_id, document_type='judgment', path=str(index), order=index, text='본문 abcdefghijklmnop', title='제목', metadata=ChunkMetadata(dataset='test', corpus_version='v1', legal_domain='civil', authority='court', decision_date=date(2024, 1, 2)))


class Retriever:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def retrieve(self, query, k, filters):
        self.calls.append((query, k, filters))
        return self.hits


def diagnose(hits=(), **kwargs):
    retriever = Retriever(list(hits))
    options = dict(gold_chunks=[chunk('gold')], gold_chunk_count=1, duplicate_question_count=1, candidate_gold_document_ids=['gold'], top_k=5, deep_k=50)
    options.update(kwargs)
    row = dict(document_id='gold', question='질문 only', answer='SECRET ANSWER', document_type='judgment', legal_domain='civil')
    return diagnose_case(row, retriever, **options), retriever


def test_duplicate_index_counts_rows_and_distinct_documents():
    rows = [{'question': '  질문\n 내용 ', 'document_id': 'b'}, {'question': '질문 내용', 'document_id': 'a'}, {'question': '질문  내용', 'document_id': 'a'}]
    assert normalize_question('  질문\n 내용 ') == '질문 내용'
    assert build_duplicate_question_index(iter(rows)) == {'질문 내용': {'count': 3, 'document_ids': ['a', 'b']}}


@pytest.mark.parametrize('rank,bucket', [(None, 'not_found'), (2, 'top_k'), (42, 'deep_candidate')])
def test_raw_chunk_gold_rank_and_question_only(rank, bucket):
    hits = [ScoredChunk(chunk=chunk('other', n), score=1.0, retriever='dense') for n in range(55)]
    if rank:
        hits[rank - 1] = ScoredChunk(chunk=chunk('gold'), score=.5, retriever='dense')
    result, retriever = diagnose(hits)
    assert result['gold_rank'] == rank
    assert result['rank_bucket'] == bucket
    assert result['gold_found_in_top_k'] is (rank is not None and rank <= 5)
    assert result['gold_found_in_deep_k'] is (rank is not None)
    assert result['exact_gold_hit'] == result['gold_found_in_top_k']
    assert len(result['retrieved']) == 50
    assert result['retrieved'][-1]['rank'] == 50
    query, depth, filters = retriever.calls[0]
    assert query == '질문 only'
    assert depth == 50
    assert filters.legal_domain == 'civil'
    assert filters.document_types is None


def test_multi_positive_is_separate_from_exact_gold():
    result, _ = diagnose([ScoredChunk(chunk=chunk('alternative'), score=.7, retriever='sparse')], duplicate_question_count=3, candidate_gold_document_ids=['gold', 'alternative'])
    assert result['multi_positive_hit'] is True
    assert result['exact_gold_hit'] is False
    assert result['duplicate_question_count'] == 3
    assert result['candidate_gold_document_ids'] == ['alternative', 'gold']


def test_missing_corpus_does_not_retrieve():
    result, retriever = diagnose(gold_chunks=[], gold_chunk_count=0)
    assert result['status'] == result['rank_bucket'] == 'missing_gold_corpus'
    assert result['gold_rank'] is None
    assert retriever.calls == []


def test_previews_metadata_scores_and_manual_fields_are_json_ready():
    result, _ = diagnose([ScoredChunk(chunk=chunk('other'), score=.2, retriever='judgment:dense')], gold_chunks=[chunk('gold', n) for n in range(4)], gold_chunk_count=8, preview_chars=4, max_gold_chunks=2)
    hit = result['retrieved'][0]
    assert hit['text_preview'] == '본문 a'
    assert hit['dense_score'] == .2
    assert 'sparse_score' not in hit and 'hybrid_score' not in hit
    assert hit['title'] == '제목' and hit['authority'] == 'court' and hit['date'] == '2024-01-02'
    assert len(result['gold_chunks']) == 2
    assert result['gold_chunks'][1]['chunk_index'] == 1
    assert result['gold_chunk_count'] == 8
    assert result['manual_category'] is None
    assert result['manual_relevant_result_ranks'] == []
    assert result['manual_notes'] == ''
    json.dumps(result)


def test_stable_id_and_override():
    first, _ = diagnose()
    second, _ = diagnose()
    assert first['case_id'] == second['case_id']
    result, _ = diagnose(case_id='specific')
    assert result['case_id'] == 'specific'


def test_rerank_score_not_mislabeled_as_hybrid():
    result, _ = diagnose([ScoredChunk(chunk=chunk('gold'), score=9.0, retriever='reranked')])
    assert result['retrieved'][0]['rerank_score'] == 9.0
    assert 'hybrid_score' not in result['retrieved'][0]


def test_trace_preserves_real_per_leg_scores_without_inventing_scores():
    retriever = Retriever([ScoredChunk(chunk=chunk('gold'), score=7.0, retriever='reranked')])
    retriever.last_trace = {'dense:question': [('gold#0', .8)], 'sparse:question': [('gold#0', 2.4)], 'fused': [('gold#0', .03)]}
    result = diagnose_case(dict(document_id='gold', question='question', answer='answer'), retriever, gold_chunks=[chunk('gold')], gold_chunk_count=1, duplicate_question_count=1, candidate_gold_document_ids=['gold'], top_k=1, deep_k=2)
    assert result['retrieval_trace'] == {'dense:question': [['gold#0', .8]], 'sparse:question': [['gold#0', 2.4]], 'fused': [['gold#0', .03]]}
    assert result['retrieved'][0]['dense_score'] == .8
    assert result['retrieved'][0]['sparse_score'] == 2.4
    assert result['retrieved'][0]['hybrid_score'] == .03
    json.dumps(result)


@pytest.mark.parametrize('limits', [{'top_k': 0}, {'deep_k': 2}, {'preview_chars': -1}, {'max_gold_chunks': -1}, {'gold_chunk_count': -1}])
def test_invalid_limits_rejected(limits):
    with pytest.raises(ValueError):
        diagnose(**limits)


def test_noop_does_not_claim_a_reranker_score():
    from legal_rag.reranking import NoopReranker
    retriever = Retriever([ScoredChunk(chunk=chunk('gold'), score=.03, retriever='reranked')])
    retriever.reranker = NoopReranker()
    retriever.last_trace = {'fused': [('gold#0', .03)]}
    result = diagnose_case(dict(document_id='gold', question='question'), retriever,
                           gold_chunks=[chunk('gold')], gold_chunk_count=1,
                           duplicate_question_count=1, candidate_gold_document_ids=['gold'], top_k=1, deep_k=2)
    assert 'rerank_score' not in result['retrieved'][0]
    assert result['retrieved'][0]['hybrid_score'] == .03


def test_query_text_cannot_impersonate_a_trace_leg():
    retriever = Retriever([ScoredChunk(chunk=chunk('gold'), score=.03, retriever='reranked')])
    query = 'question:dense:value'
    retriever.last_trace = {'v1:sparse:' + query: [('gold#0', 2.4)]}
    result = diagnose_case(dict(document_id='gold', question=query), retriever,
                           gold_chunks=[chunk('gold')], gold_chunk_count=1,
                           duplicate_question_count=1, candidate_gold_document_ids=['gold'], top_k=1, deep_k=2)
    assert result['retrieved'][0]['sparse_score'] == 2.4
    assert 'dense_score' not in result['retrieved'][0]

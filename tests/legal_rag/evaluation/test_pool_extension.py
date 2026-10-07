import pytest

from legal_rag.evaluation.pool_extension import extend_pool, gold_document_ids
from legal_rag.schemas import Chunk, ChunkMetadata, DocumentType


def make_chunk(chunk_id, document_id=None, text='본문'):
    return Chunk(id=chunk_id, document_id=document_id or chunk_id.split('#')[0], document_type=DocumentType.judgment,
                 path='p', order=0, text=text, title='제목',
                 metadata=ChunkMetadata(dataset='d', corpus_version='ip-v1', authority='특허법원'))

SOURCE = [
    {'case_id': 'c1', 'chunk_id': 'a#0', 'question': '질문', 'question_domain': None, 'reference_answer': '답변', 'document_id': 'a',
     'document_type': 'judgment', 'title': '제목', 'text': '본문', 'authority': '특허법원', 'date': None,
     'case_number': 'a', 'search_query': '특허법원 a', 'ranks': {'dense': 1, 'hybrid': 2}, 'is_gold_chunk': False},
    {'case_id': 'c1', 'chunk_id': 'gold#0', 'question': '질문', 'question_domain': None, 'reference_answer': '답변', 'document_id': 'gold',
     'document_type': 'judgment', 'title': '제목', 'text': '본문', 'authority': '특허법원', 'date': None,
     'case_number': 'gold', 'search_query': '특허법원 gold', 'ranks': {}, 'is_gold_chunk': True},
]


def test_gold_document_ids_from_the_pools_own_gold_rows():
    assert gold_document_ids(SOURCE) == {'c1': {'gold'}}


def test_gold_document_ids_prefers_the_target_file_which_covers_every_case():
    target = [{'id': 'c1', 'relevant_document_ids': ['gold', 'gold2']}, {'id': 'c2', 'relevant_document_ids': ['g']}]
    assert gold_document_ids(SOURCE, target) == {'c1': {'gold', 'gold2'}, 'c2': {'g'}}


def test_gold_document_ids_rejects_a_target_that_disagrees_with_the_pools_gold_rows():
    with pytest.raises(ValueError, match='gold'):
        gold_document_ids(SOURCE, [{'id': 'c1', 'relevant_document_ids': ['other']}])


def test_source_rows_keep_their_fields_and_take_the_new_runs_ranks():
    pool = extend_pool(SOURCE, {('c1', 'a#0'): {'hybrid_s0.50': 3}}, {}, gold_documents={'c1': {'gold'}})
    assert [row['chunk_id'] for row in pool] == ['a#0', 'gold#0']
    assert pool[0] == dict(SOURCE[0], ranks={'hybrid_s0.50': 3})
    assert pool[1]['ranks'] == {}                      # an unretrieved gold chunk keeps empty ranks


def test_newly_surfaced_chunks_become_rows_carrying_the_cases_question_and_answer():
    needed = {('c1', 'a#0'): {'dense': 1}, ('c1', 'new#0'): {'hybrid_s1.00': 4}}
    pool = extend_pool(SOURCE, needed, {'new#0': make_chunk('new#0', 'new', text='새 본문')},
                       gold_documents={'c1': {'gold'}})
    new = pool[-1]
    assert [row['chunk_id'] for row in pool] == ['a#0', 'gold#0', 'new#0']
    assert new['case_id'] == 'c1' and new['question'] == '질문' and new['reference_answer'] == '답변'
    assert new['document_id'] == 'new' and new['document_type'] == 'judgment' and new['text'] == '새 본문'
    assert new['authority'] == '특허법원' and new['search_query'] == '특허법원 new'
    assert new['ranks'] == {'hybrid_s1.00': 4} and new['is_gold_chunk'] is False
    assert set(new) == set(SOURCE[0])


def test_a_new_chunk_of_the_cases_gold_document_is_flagged_gold():
    pool = extend_pool(SOURCE, {('c1', 'gold#1'): {'dense': 9}}, {'gold#1': make_chunk('gold#1', 'gold')},
                       gold_documents={'c1': {'gold'}})
    assert pool[-1]['chunk_id'] == 'gold#1' and pool[-1]['is_gold_chunk'] is True


def test_missing_chunk_text_raises_rather_than_creating_an_unjudgeable_pair():
    with pytest.raises(ValueError, match='new#0'):
        extend_pool(SOURCE, {('c1', 'new#0'): {'dense': 1}}, {}, gold_documents={})


def test_a_pair_for_a_case_the_source_pool_does_not_know_raises():
    with pytest.raises(ValueError, match='c9'):
        extend_pool(SOURCE, {('c9', 'new#0'): {'dense': 1}}, {'new#0': make_chunk('new#0')}, gold_documents={})

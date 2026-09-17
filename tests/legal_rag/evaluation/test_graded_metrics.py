import math

import pytest

from legal_rag.evaluation.graded_metrics import case_metrics, per_case_metrics, summarize_runs


def ranked(*pairs):
    return [{'chunk_id': c, 'document_id': d} for c, d in pairs]


def test_graded_ndcg_success_mrr_and_coverage():
    grades = {'a': 0, 'b': 2, 'c': 1, 'z': 2}               # z judged but not retrieved
    result = case_metrics(ranked(('a', 'A'), ('b', 'B'), ('x', 'X'), ('c', 'C')), grades, level='chunk')
    dcg = 3 / math.log2(3) + 1 / math.log2(5)
    idcg = 3 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)
    assert result['ndcg@10'] == pytest.approx(dcg / idcg)
    assert result['success@5'] == 1 and result['mrr'] == pytest.approx(1 / 2)
    assert result['precision@10_grade2'] == pytest.approx(1 / 10)
    assert result['judged_coverage@10'] == pytest.approx(3 / 4)


def test_document_level_takes_first_chunk_rank_and_max_grade():
    grades = {'a1': 0, 'a2': 2, 'b1': 1, 'c9': 2}
    documents = {'a1': 'A', 'a2': 'A', 'b1': 'B', 'c9': 'C'}   # c9 judged via another run, not retrieved here
    result = case_metrics(ranked(('a1', 'A'), ('b1', 'B')), grades, level='document', documents=documents)
    assert result['mrr'] == 1.0                                # document A = max(0, 2) at rank 1
    assert result['judged_coverage@10'] == 1.0
    ideal = 3 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)   # A=2, C=2, B=1
    assert result['ndcg@10'] == pytest.approx((3 / math.log2(2) + 1 / math.log2(3)) / ideal)


def test_case_without_relevant_chunks_has_no_ndcg():
    result = case_metrics(ranked(('a', 'A')), {'a': 0}, level='chunk')
    assert result['ndcg@10'] is None and result['mrr'] == 0.0


def test_summary_means_skip_undefined_ndcg_and_report_counts():
    runs = {'dense': [{'case_id': 'c1', 'ranked': ranked(('a', 'A'))},
                      {'case_id': 'c2', 'ranked': ranked(('b', 'B'))}]}
    grades = {('c1', 'a'): 2, ('c2', 'b'): 0}
    summary = summarize_runs(runs, grades, {('c1', 'a'): 'A', ('c2', 'b'): 'B'})
    chunk = summary['dense']['chunk']
    assert chunk['cases'] == 2 and chunk['cases_without_relevant'] == 1
    assert chunk['ndcg@10'] == 1.0 and chunk['success@5'] == 0.5
    assert chunk['min_judged_coverage@10'] == pytest.approx(1.0)


def test_mode_with_no_cases_raises():
    with pytest.raises(ValueError, match='dense'):
        summarize_runs({'dense': []}, {}, {})


def test_modes_over_different_case_sets_raise():
    runs = {'dense': [{'case_id': 'c1', 'ranked': ranked(('a', 'A'))}],
            'bm25': [{'case_id': 'c2', 'ranked': ranked(('a', 'A'))}]}
    with pytest.raises(ValueError, match='same cases'):
        summarize_runs(runs, {('c1', 'a'): 2}, {('c1', 'a'): 'A'})


def test_per_case_metrics_rows_carry_case_id():
    runs = {'dense': [{'case_id': 'c1', 'ranked': ranked(('a', 'A'), ('b', 'A'))}]}
    grades = {('c1', 'a'): 1, ('c1', 'b'): 2}
    per_case = per_case_metrics(runs, grades, {('c1', 'a'): 'A', ('c1', 'b'): 'A'})
    (chunk_row,) = per_case['dense']['chunk']
    (doc_row,) = per_case['dense']['document']
    assert chunk_row['case_id'] == 'c1' and chunk_row['mrr'] == pytest.approx(1 / 2)
    assert doc_row['case_id'] == 'c1' and doc_row['mrr'] == 1.0


def test_document_coverage_counts_document_judged_via_other_chunk():
    # Retrieved chunk a2 is unjudged, but a1 of the same document was judged grade 0:
    # the document counts as judged (grade 0) while chunk level flags the gap.
    grades = {'a1': 0}
    documents = {'a1': 'A'}
    hits = ranked(('a2', 'A'))
    assert case_metrics(hits, grades, level='document', documents=documents)['judged_coverage@10'] == 1.0
    assert case_metrics(hits, grades, level='chunk')['judged_coverage@10'] == 0.0

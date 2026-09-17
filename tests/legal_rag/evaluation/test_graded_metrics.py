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
    assert result['success@5'] == 1 and result['mrr@20'] == pytest.approx(1 / 2)
    assert result['precision@10_grade2'] == pytest.approx(1 / 10)
    assert result['judged_coverage@10'] == pytest.approx(3 / 4)


def test_document_level_takes_first_chunk_rank_and_max_grade():
    grades = {'a1': 0, 'a2': 2, 'b1': 1, 'c9': 2}
    documents = {'a1': 'A', 'a2': 'A', 'b1': 'B', 'c9': 'C'}   # c9 judged via another run, not retrieved here
    result = case_metrics(ranked(('a1', 'A'), ('b1', 'B')), grades, level='document', documents=documents)
    assert result['mrr@20'] == 1.0                                # document A = max(0, 2) at rank 1
    assert result['judged_coverage@10'] == 1.0
    ideal = 3 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)   # A=2, C=2, B=1
    assert result['ndcg@10'] == pytest.approx((3 / math.log2(2) + 1 / math.log2(3)) / ideal)


def test_case_without_relevant_chunks_has_no_ndcg():
    result = case_metrics(ranked(('a', 'A')), {'a': 0}, level='chunk')
    assert result['ndcg@10'] is None and result['mrr@20'] == 0.0


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
    assert chunk_row['case_id'] == 'c1' and chunk_row['mrr@20'] == pytest.approx(1 / 2)
    assert doc_row['case_id'] == 'c1' and doc_row['mrr@20'] == 1.0


def test_document_coverage_counts_document_judged_via_other_chunk():
    # Retrieved chunk a2 is unjudged, but a1 of the same document was judged grade 0:
    # the document counts as judged (grade 0) while chunk level flags the gap.
    grades = {'a1': 0}
    documents = {'a1': 'A'}
    hits = ranked(('a2', 'A'))
    assert case_metrics(hits, grades, level='document', documents=documents)['judged_coverage@10'] == 1.0
    assert case_metrics(hits, grades, level='chunk')['judged_coverage@10'] == 0.0


def test_failed_chunk_counts_as_covered_with_zero_gain_and_is_reported():
    grades = {'a': 2}
    hits = ranked(('x', 'X'), ('a', 'A'))
    result = case_metrics(hits, grades, level='chunk', failed={'x'})
    assert result['judged_coverage@10'] == 1.0
    assert result['failed@10'] == 1
    assert result['ndcg@10'] == pytest.approx((3 / math.log2(3)) / 3)   # x has gain 0, not in ideal
    assert result['mrr@20'] == pytest.approx(1 / 2)


def test_document_judged_via_failed_chunk_only_counts_as_failed():
    documents = {'x1': 'X', 'a1': 'A'}
    result = case_metrics(ranked(('x1', 'X'), ('a1', 'A')), {'a1': 1}, level='document',
                          documents=documents, failed={'x1'})
    assert result['judged_coverage@10'] == 1.0 and result['failed@10'] == 1
    graded_too = case_metrics(ranked(('x1', 'X')), {'x2': 2}, level='document',
                              documents={'x1': 'X', 'x2': 'X'}, failed={'x1'})
    assert graded_too['failed@10'] == 0                          # X has a graded chunk


def test_summary_totals_failed_top10_from_store_failures():
    runs = {'dense': [{'case_id': 'c1', 'ranked': ranked(('a', 'A'), ('b', 'B'))}]}
    summary = summarize_runs(runs, {('c1', 'a'): 2}, {('c1', 'a'): 'A', ('c1', 'b'): 'B'},
                             failures={('c1', 'b')})
    chunk = summary['dense']['chunk']
    assert chunk['min_judged_coverage@10'] == 1.0 and chunk['failed_top10_total'] == 1
    assert summary['dense']['document']['failed_top10_total'] == 1
    per_case = per_case_metrics(runs, {('c1', 'a'): 2}, {('c1', 'a'): 'A', ('c1', 'b'): 'B'},
                                failures={('c1', 'b')})
    assert per_case['dense']['chunk'][0]['failed@10'] == 1


def test_mrr_and_success_are_capped_at_the_judged_depth():
    hits = ranked(*[(f'k{i}', f'D{i}') for i in range(1, 26)])
    grades = {f'k{i}': 0 for i in range(1, 26)} | {'k25': 2}
    deep = case_metrics(hits, grades, level='chunk')
    assert deep['mrr@20'] == 0.0 and 'mrr' not in deep          # rank 25 is beyond pool depth 20
    shallow = case_metrics(hits, {'k3': 2}, level='chunk', depth=10)
    assert shallow['mrr@10'] == pytest.approx(1 / 3)
    assert 'success@20' not in shallow and 'judged_coverage@20' not in shallow
    assert shallow['success@10'] == 1


def test_coverage_at_20_is_reported_and_summarized():
    hits = ranked(*[(f'k{i}', f'D{i}') for i in range(1, 21)])
    grades = {f'k{i}': 0 for i in range(1, 16)}
    result = case_metrics(hits, grades, level='chunk')
    assert result['judged_coverage@10'] == 1.0 and result['judged_coverage@20'] == pytest.approx(15 / 20)
    summary = summarize_runs({'dense': [{'case_id': 'c1', 'ranked': hits}]},
                             {('c1', k): g for k, g in grades.items()},
                             {('c1', k): 'D' + k[1:] for k in grades}, depth=20)
    chunk = summary['dense']['chunk']
    assert chunk['min_judged_coverage@20'] == pytest.approx(0.75) and chunk['mrr@20'] == 0.0
    shallow = summarize_runs({'dense': [{'case_id': 'c1', 'ranked': hits}]},
                             {('c1', k): g for k, g in grades.items()},
                             {('c1', k): 'D' + k[1:] for k in grades}, depth=10)['dense']['chunk']
    assert 'min_judged_coverage@20' not in shallow and 'mrr@10' in shallow


def test_coverage_cutoffs_never_exceed_a_shallow_pool_depth():
    hits = ranked(*[(f'k{i}', f'D{i}') for i in range(1, 11)])
    result = case_metrics(hits, {f'k{i}': 0 for i in range(1, 6)}, level='chunk', depth=5)
    assert result['judged_coverage@5'] == 1.0 and 'judged_coverage@10' not in result


def test_threshold_1_counts_grade_1_as_relevant_for_headline_metrics():
    grades = {'a': 0, 'b': 1, 'c': 2, 'z': 1}                # z judged but not retrieved
    hits = ranked(('a', 'A'), ('b', 'B'), ('x', 'X'), ('c', 'C'))
    strict = case_metrics(hits, grades, level='chunk')         # default threshold 2
    loose = case_metrics(hits, grades, level='chunk', threshold=1)
    assert strict['mrr@20'] == pytest.approx(1 / 4) and loose['mrr@20'] == pytest.approx(1 / 2)
    assert strict['precision@10'] == pytest.approx(1 / 10) and loose['precision@10'] == pytest.approx(2 / 10)
    assert strict['success@5'] == loose['success@5'] == 1
    assert case_metrics(ranked(('a', 'A'), ('b', 'B')), grades, level='chunk')['success@5'] == 0
    assert case_metrics(ranked(('a', 'A'), ('b', 'B')), grades, level='chunk', threshold=1)['success@5'] == 1
    # binary nDCG: gain 1 at grade >= threshold; ideal from every judged chunk at/above it
    assert loose['ndcg@10_binary'] == pytest.approx(
        (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3) + 1 / math.log2(4)))
    assert strict['ndcg@10_binary'] == pytest.approx(1 / math.log2(5))
    assert loose['ndcg@10'] == strict['ndcg@10']               # graded nDCG ignores the threshold
    assert loose['precision@10_grade2'] == strict['precision@10_grade2'] == pytest.approx(1 / 10)
    with pytest.raises(ValueError):
        case_metrics(hits, grades, level='chunk', threshold=3)


def test_threshold_at_document_level_uses_max_grade_per_document():
    grades = {'a1': 0, 'a2': 1, 'b1': 0, 'c9': 1}
    documents = {'a1': 'A', 'a2': 'A', 'b1': 'B', 'c9': 'C'}
    hits = ranked(('b1', 'B'), ('a1', 'A'))
    result = case_metrics(hits, grades, level='document', documents=documents, threshold=1)
    assert result['mrr@20'] == pytest.approx(1 / 2) and result['success@5'] == 1
    assert result['precision@10'] == pytest.approx(1 / 10)
    assert result['ndcg@10_binary'] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))
    strict = case_metrics(hits, grades, level='document', documents=documents)
    assert strict['mrr@20'] == 0.0 and strict['ndcg@10_binary'] is None and strict['ndcg@10'] is not None


def test_summary_skips_undefined_binary_ndcg_and_counts_those_cases():
    runs = {'dense': [{'case_id': 'c1', 'ranked': ranked(('a', 'A'))},
                      {'case_id': 'c2', 'ranked': ranked(('b', 'B'))}]}
    grades, documents = {('c1', 'a'): 2, ('c2', 'b'): 1}, {('c1', 'a'): 'A', ('c2', 'b'): 'B'}
    strict = summarize_runs(runs, grades, documents)['dense']['chunk']
    assert strict['ndcg@10_binary'] == 1.0 and strict['cases_without_relevant_binary'] == 1
    assert strict['cases_without_relevant'] == 0 and strict['precision@10'] == pytest.approx(0.05)
    loose = summarize_runs(runs, grades, documents, threshold=1)['dense']['chunk']
    assert loose['cases_without_relevant_binary'] == 0 and loose['success@5'] == 1.0
    assert loose['precision@10'] == pytest.approx(0.1)
    per_case = per_case_metrics(runs, grades, documents, threshold=1)
    assert per_case['dense']['document'][1]['mrr@20'] == 1.0


def test_restrict_to_pool_drops_out_of_pool_judgments_and_counts_them():
    from legal_rag.evaluation.graded_metrics import restrict_to_pool
    grades = {('c1', 'a'): 1, ('c1', 'z'): 2, ('c2', 'b'): 0}
    documents = {('c1', 'a'): 'A', ('c1', 'z'): 'Z', ('c2', 'b'): 'B', ('c1', 'f'): 'F', ('c1', 'g'): 'G'}
    failures = {('c1', 'f'), ('c1', 'g')}
    pool_pairs = {('c1', 'a'), ('c2', 'b'), ('c1', 'f')}
    kept_grades, kept_documents, kept_failures, ignored = restrict_to_pool(grades, documents, failures, pool_pairs)
    assert kept_grades == {('c1', 'a'): 1, ('c2', 'b'): 0}
    assert kept_documents == {('c1', 'a'): 'A', ('c2', 'b'): 'B', ('c1', 'f'): 'F'}
    assert kept_failures == {('c1', 'f')} and ignored == 2          # z graded, g failed


def test_out_of_pool_grade2_judgment_no_longer_lowers_ndcg():
    from legal_rag.evaluation.graded_metrics import restrict_to_pool
    runs = {'dense': [{'case_id': 'c1', 'ranked': ranked(('a', 'A'), ('b', 'B'))}]}
    grades = {('c1', 'a'): 1, ('c1', 'b'): 0, ('c1', 'deep'): 2}     # 'deep' was judged for a deeper pool
    documents = {('c1', 'a'): 'A', ('c1', 'b'): 'B', ('c1', 'deep'): 'D'}
    polluted = summarize_runs(runs, grades, documents, threshold=1)['dense']
    assert polluted['chunk']['ndcg@10'] < 1 and polluted['chunk']['ndcg@10_binary'] < 1
    kept = restrict_to_pool(grades, documents, set(), {('c1', 'a'), ('c1', 'b')})
    for threshold in (1, 2):
        summary = summarize_runs(runs, *kept[:3], threshold=threshold)['dense']
        for level in ('chunk', 'document'):
            assert summary[level]['ndcg@10'] == 1.0
    assert summarize_runs(runs, *kept[:3], threshold=1)['dense']['chunk']['ndcg@10_binary'] == 1.0

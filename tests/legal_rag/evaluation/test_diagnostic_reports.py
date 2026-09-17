import csv
import json

import pytest

from legal_rag.evaluation.reports import summarize_cases, write_diagnostic_reports


def case(rank, *, missing=False):
    return dict(case_id=str(rank), question='질문, "인용"\n두 줄', answer='답변',
                gold_document_id='gold', gold_rank=rank, gold_found_in_top_k=rank == 1,
                gold_found_in_deep_k=rank is not None, exact_gold_hit=rank == 1,
                multi_positive_hit=rank in (1, 42), duplicate_question_count=2,
                status='missing_gold_corpus' if missing else 'evaluated', retrieved=[],
                manual_category=None, manual_relevant_result_ranks=[], manual_notes='')


def test_summary_excludes_missing_and_uses_all_evaluated_not_report_sample():
    result = summarize_cases([case(1), case(42), case(None), case(None, missing=True)],
                             legal_domain='civil', retrieval_mode='hybrid', top_k=20, deep_k=100)
    assert result['evaluated'] == 3
    assert result['missing_gold_corpus'] == 1
    assert result['gold_rank_le_top_k'] == 1
    assert result['gold_rank_between_top_k_and_deep_k'] == 1
    assert result['gold_not_found_within_deep_k'] == 1
    assert result['exact_recall_at_k'] == pytest.approx(1/3)
    assert result['multi_positive_recall_at_k'] == pytest.approx(2/3)
    assert result['recall_at']['100'] == pytest.approx(2/3)
    assert result['mrr'] == pytest.approx((1 + 1/42)/3)
    assert result['gold_rank_le_top_k_percentage'] == pytest.approx(100/3)


def test_no_claim_for_recall_beyond_search_depth():
    result = summarize_cases([], legal_domain='civil', retrieval_mode='dense', top_k=5, deep_k=10)
    assert '20' not in result['recall_at']
    assert result['exact_recall_at_k'] is None


def test_reports_round_trip_and_never_overwrite_reviews(tmp_path):
    output = tmp_path / 'report'
    record = case(None)
    write_diagnostic_reports(output, [record], [case(None, missing=True)], {'evaluated': 1})
    assert json.loads((output/'failures.jsonl').read_text()) == record
    with (output/'review.csv').open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]['question'] == record['question']
    assert 'top_3_text_preview' in rows[0]
    assert 'top_4_text_preview' not in rows[0]
    (output/'review.csv').write_text('human annotation')
    with pytest.raises(FileExistsError):
        write_diagnostic_reports(output, [record], [], {})
    assert (output/'review.csv').read_text() == 'human annotation'

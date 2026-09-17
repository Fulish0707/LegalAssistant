import csv

import pytest

from legal_rag.evaluation.graded_metrics import per_case_metrics, summarize_runs
from legal_rag.evaluation.graded_report import (
    case_rows, min_chunk_coverage, render_markdown, write_cases_csv)

RUNS = {'dense': [{'case_id': 'c1', 'ranked': [{'chunk_id': 'k1', 'document_id': 'd1'},
                                               {'chunk_id': 'k2', 'document_id': 'd2'}]}],
        'sparse': [{'case_id': 'c1', 'ranked': [{'chunk_id': 'k2', 'document_id': 'd2'}]}]}
JUDGE = {'model': 'ollama:qwen3:8b', 'prompt_sha256': 'abcdef0123456789', 'failed_parses': 0}
POOL = {'corpus_version': 'ip-v1', 'filters': {'legal_domain': 'intellectual_property'},
        'pool_depth': 20, 'settings_hash': 'h1'}


def test_markdown_has_provenance_all_rows_and_no_warning_when_fully_judged():
    grades = {('c1', 'k1'): 2, ('c1', 'k2'): 0}
    summary = summarize_runs(RUNS, grades, {('c1', 'k1'): 'd1', ('c1', 'k2'): 'd2'})
    text = render_markdown(summary, judge=JUDGE, pool=POOL)
    assert 'abcdef012345' in text and 'settings_hash `h1`' in text
    assert 'Scoped to ip-v1 with filter' in text and 'multi-corpus fan-out is not measured' in text
    assert '| cases_without_relevant | 0 | 0 |' in text and '| mean_judged_coverage@10 |' in text
    assert '| ndcg@10 | 1.000 | 0.000 |' in text and '| success@5 | 1.000 | 0.000 |' in text
    assert 'WARNING' not in text and min_chunk_coverage(summary) == 1


def test_markdown_warns_when_top10_unjudged():
    summary = summarize_runs(RUNS, {('c1', 'k1'): 2}, {('c1', 'k1'): 'd1'})
    assert min_chunk_coverage(summary) == pytest.approx(0.0)
    assert 'WARNING' in render_markdown(summary, judge=JUDGE, pool=POOL)


def test_case_rows_flatten_modes_and_levels(tmp_path):
    grades = {('c1', 'k1'): 2}
    fields, rows = case_rows(per_case_metrics(RUNS, grades, {('c1', 'k1'): 'd1'}), {'c1': '질문'})
    assert fields[:2] == ['case_id', 'question'] and 'sparse.document.judged_coverage@10' in fields
    assert len(rows) == 1 and rows[0]['question'] == '질문'
    assert rows[0]['dense.chunk.mrr'] == 1.0 and rows[0]['sparse.chunk.success@10'] == 0
    write_cases_csv(tmp_path / 'cases.csv', fields, rows)
    with (tmp_path / 'cases.csv').open(newline='', encoding='utf-8-sig') as handle:
        assert list(csv.DictReader(handle))[0]['case_id'] == 'c1'
    with pytest.raises(FileExistsError):
        write_cases_csv(tmp_path / 'cases.csv', fields, rows)

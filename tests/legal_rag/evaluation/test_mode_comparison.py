import csv
import json

import pytest

from legal_rag.evaluation.diagnosis.mode_comparison import compare_mode_runs, write_mode_comparison

TARGET = [{'id': 'qa:a', 'query': '질문 A', 'relevant_document_ids': ['doc-a']},
          {'id': 'qa:b', 'query': '질문 B', 'relevant_document_ids': ['doc-b']},
          {'id': 'qa:c', 'query': '질문 C', 'relevant_document_ids': ['doc-c']}]


def run_dir(tmp_path, mode, ranks, *, target=TARGET, settings_hash='h1', name=None):
    directory = tmp_path / (name or f'{mode}-run')
    directory.mkdir()
    (directory / 'target.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in target), encoding='utf-8')
    (directory / 'evaluated.jsonl').write_text(''.join(
        json.dumps({'case_id': case_id, 'status': 'ok', 'gold_rank': rank, 'multi_positive_hit': False,
                    'gold_found_in_top_k': False, 'duplicate_question_count': 1}) + '\n'
        for case_id, rank in ranks.items()), encoding='utf-8')
    (directory / 'summary.json').write_text(json.dumps({
        'retrieval_mode': mode, 'legal_domain': 'intellectual_property', 'corpus_version': 'ip-v1',
        'top_k': 20, 'deep_k': 100, 'settings_hash': settings_hash, 'seed': 42}), encoding='utf-8')
    return directory


def three_runs(tmp_path):
    return [run_dir(tmp_path, 'dense', {'qa:a': 3, 'qa:b': None, 'qa:c': None}),
            run_dir(tmp_path, 'sparse', {'qa:a': None, 'qa:b': 42, 'qa:c': None}),
            run_dir(tmp_path, 'hybrid', {'qa:a': 7, 'qa:b': 25, 'qa:c': None})]


def test_metrics_per_mode_over_the_shared_target(tmp_path):
    report = compare_mode_runs(three_runs(tmp_path))
    hybrid = report['modes']['hybrid']
    assert report['target_cases'] == 3
    assert list(report['modes']) == ['dense', 'sparse', 'hybrid']
    assert hybrid['recall_at'] == {'5': 0, '10': pytest.approx(1/3), '20': pytest.approx(1/3),
                                   '50': pytest.approx(2/3), '100': pytest.approx(2/3)}
    assert hybrid['mrr'] == pytest.approx((1/7 + 1/25) / 3)
    assert hybrid['buckets'] == {'top_20': 1, 'rank_21_100': 1, 'not_found': 1}


def test_cases_carry_every_modes_gold_rank(tmp_path):
    cases = {case['case_id']: case for case in compare_mode_runs(three_runs(tmp_path))['cases']}
    assert cases['qa:b'] == {'case_id': 'qa:b', 'query': '질문 B', 'gold_document_ids': ['doc-b'],
                             'dense': None, 'sparse': 42, 'hybrid': 25}


def test_runs_over_different_targets_are_refused(tmp_path):
    runs = three_runs(tmp_path)
    runs[1] = run_dir(tmp_path, 'sparse', {'qa:a': 1, 'qa:b': 1}, target=TARGET[:2], name='short')
    with pytest.raises(ValueError, match='target'):
        compare_mode_runs(runs)


def test_runs_under_different_settings_are_refused(tmp_path):
    runs = three_runs(tmp_path)
    runs[2] = run_dir(tmp_path, 'hybrid', {'qa:a': 1, 'qa:b': 1, 'qa:c': 1}, settings_hash='h2', name='other')
    with pytest.raises(ValueError, match='settings'):
        compare_mode_runs(runs)


def test_writes_json_markdown_and_case_csv_without_overwriting(tmp_path):
    report = compare_mode_runs(three_runs(tmp_path))
    output = tmp_path / 'comparison'
    write_mode_comparison(output, report)
    assert json.loads((output / 'comparison.json').read_text())['target_cases'] == 3
    markdown = (output / 'comparison.md').read_text()
    assert 'Recall@5' in markdown and 'MRR' in markdown and 'Not found' in markdown
    with (output / 'cases.csv').open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1]['sparse_rank'] == '42' and rows[1]['dense_rank'] == ''
    with pytest.raises(FileExistsError):
        write_mode_comparison(output, report)

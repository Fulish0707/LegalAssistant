import csv

import pytest

from legal_rag.evaluation.graded_metrics import per_case_metrics, summarize_runs
from legal_rag.evaluation.graded_report import (
    case_rows, failed_top10_total, min_chunk_coverage, render_markdown, write_cases_csv)

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
    assert rows[0]['dense.chunk.mrr@20'] == 1.0 and rows[0]['sparse.chunk.success@10'] == 0
    write_cases_csv(tmp_path / 'cases.csv', fields, rows)
    with (tmp_path / 'cases.csv').open(newline='', encoding='utf-8-sig') as handle:
        assert list(csv.DictReader(handle))[0]['case_id'] == 'c1'
    with pytest.raises(FileExistsError):
        write_cases_csv(tmp_path / 'cases.csv', fields, rows)


def test_markdown_reports_failed_top10_counts():
    summary = summarize_runs(RUNS, {('c1', 'k1'): 2}, {('c1', 'k1'): 'd1', ('c1', 'k2'): 'd2'},
                             failures={('c1', 'k2')})
    text = render_markdown(summary, judge=dict(JUDGE, failed_parses=1), pool=POOL)
    assert '| failed_top10_total | 1 | 1 |' in text
    assert min_chunk_coverage(summary) == 1 and failed_top10_total(summary) == 2


def test_validity_uses_coverage_at_20_and_rows_follow_depth():
    hits = [{'chunk_id': f'k{i}', 'document_id': f'd{i}'} for i in range(1, 21)]
    runs = {'dense': [{'case_id': 'c1', 'ranked': hits}]}
    grades = {('c1', f'k{i}'): 0 for i in range(1, 11)}
    documents = {key: 'd' + key[1][1:] for key in grades}
    summary = summarize_runs(runs, grades, documents, depth=20)
    assert summary['dense']['chunk']['min_judged_coverage@10'] == 1.0
    assert min_chunk_coverage(summary) == pytest.approx(0.5)
    text = render_markdown(summary, judge=JUDGE, pool=POOL)
    assert '| mrr@20 |' in text and '| min_judged_coverage@20 |' in text and 'WARNING' in text
    shallow = summarize_runs(runs, grades, documents, depth=10)
    text = render_markdown(shallow, judge=JUDGE, pool=dict(POOL, pool_depth=10))
    assert '| mrr@10 |' in text and 'success@20' not in text and min_chunk_coverage(shallow) == 1
    fields, _ = case_rows(per_case_metrics(runs, grades, documents, depth=10), {})
    assert 'dense.chunk.mrr@10' in fields and 'dense.chunk.failed@10' in fields


def test_markdown_at_threshold_1_moves_graded_rows_to_reference_block():
    grades = {('c1', 'k1'): 1, ('c1', 'k2'): 0}
    documents = {('c1', 'k1'): 'd1', ('c1', 'k2'): 'd2'}
    summary = summarize_runs(RUNS, grades, documents, threshold=1)
    text = render_markdown(summary, judge=JUDGE, pool=POOL, threshold=1)
    assert '- Headline relevance: grade ≥ 1' in text
    chunk = text.split('## chunk')[1].split('## document')[0]
    headline, reference = chunk.split('### Reference (graded)')
    assert '| success@5 | 1.000 | 0.000 |' in headline and '| ndcg@10_binary | 1.000 | 0.000 |' in headline
    assert '| precision@10 |' in headline and '| ndcg@10 |' not in headline
    assert '| cases_without_relevant |' not in headline and '| cases_without_relevant |' in reference
    assert '| cases_without_relevant_binary |' in headline
    assert '| ndcg@10 | 1.000 | 0.000 |' in reference and '| precision@10_grade2 |' in reference
    assert '| success@5 |' not in reference
    default = render_markdown(summarize_runs(RUNS, grades, documents), judge=JUDGE, pool=POOL)
    assert '- Headline relevance: grade ≥ 2' in default and 'Reference (graded)' not in default
    assert '| ndcg@10 | 1.000 | 0.000 |' in default and '| ndcg@10_binary | — | — |' in default


def test_case_rows_record_threshold_and_binary_metrics():
    grades = {('c1', 'k1'): 1}
    per_case = per_case_metrics(RUNS, grades, {('c1', 'k1'): 'd1'}, threshold=1)
    fields, rows = case_rows(per_case, {'c1': '질문'}, threshold=1)
    assert fields[:3] == ['case_id', 'question', 'relevance_threshold'] and rows[0]['relevance_threshold'] == 1
    assert 'dense.chunk.ndcg@10_binary' in fields and 'dense.chunk.precision@10' in fields
    assert rows[0]['dense.chunk.success@10'] == 1


def test_eval_graded_script_threshold_is_recorded_in_every_output(tmp_path):
    import importlib.util
    import json
    from pathlib import Path
    from typer.testing import CliRunner
    from legal_rag.evaluation.judgment_store import JudgmentStore
    from legal_rag.evaluation.relevance_judge import judge_key
    pool_dir = tmp_path / 'pool'
    (pool_dir / 'runs').mkdir(parents=True)
    for mode, rows in RUNS.items():
        (pool_dir / 'runs' / f'{mode}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (pool_dir / 'pool.json').write_text(json.dumps(dict(POOL, pool_depth=10)))
    (pool_dir / 'pool.jsonl').write_text(''.join(json.dumps({'case_id': 'c1', 'chunk_id': chunk, 'question': '질문'})
                                                 + '\n' for chunk in ('k1', 'k2')))
    store = JudgmentStore(pool_dir / 'judgments.jsonl')
    key = judge_key('ollama:qwen3:8b', think=True)
    store.append(key, case_id='c1', chunk_id='k1', document_id='d1', grade=0, reason='r')
    store.append(key, case_id='c1', chunk_id='k2', document_id='d2', grade=1, reason='r')
    store.append(key, case_id='c1', chunk_id='deep', document_id='d9', grade=2, reason='out of pool')
    path = Path(__file__).resolve().parents[3] / 'scripts' / 'eval_graded.py'
    spec = importlib.util.spec_from_file_location('script_eval_graded', path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    out = tmp_path / 'out'
    result = CliRunner().invoke(script.app, ['--pool-dir', str(pool_dir), '--think', '--threshold', '1',
                                             '--output', str(out)])
    assert result.exit_code == 0, result.output
    report = json.loads((out / 'graded.json').read_text(encoding='utf-8'))
    assert report['threshold'] == 1 and report['modes']['sparse']['chunk']['success@5'] == 1.0
    assert report['judge']['out_of_pool_judgments'] == 1
    assert report['modes']['sparse']['chunk']['ndcg@10_binary'] == 1.0     # 'deep' grade 2 not in the ideal
    markdown = (out / 'graded.md').read_text(encoding='utf-8')
    assert 'Headline relevance: grade ≥ 1' in markdown and 'out-of-pool judgments ignored: 1' in markdown
    with (out / 'cases.csv').open(newline='', encoding='utf-8-sig') as handle:
        assert list(csv.DictReader(handle))[0]['relevance_threshold'] == '1'
    default = CliRunner().invoke(script.app, ['--pool-dir', str(pool_dir), '--think'])
    assert default.exit_code == 0 and 'Headline relevance: grade ≥ 2' in default.output
    assert CliRunner().invoke(script.app, ['--pool-dir', str(pool_dir), '--threshold', '3']).exit_code != 0


def test_the_scope_line_describes_a_fan_out_when_the_pool_records_one():
    from legal_rag.evaluation.graded_report import _scope
    assert _scope({'corpus_version': 'ip-v1', 'filters': {'legal_domain': 'intellectual_property'}}) == (
        "- Scoped to ip-v1 with filter {'legal_domain': 'intellectual_property'}; "
        'multi-corpus fan-out is not measured.')
    line = _scope({'filters': None, 'runs_corpora': {'ip_only': ['ip-v1'],
                                                     'fanout': ['criminal-v1', 'ip-v1'], 'oracle': 'own'}})
    assert line == ('- Corpora per run: `ip_only` ip-v1, `fanout` criminal-v1 + ip-v1, '
                    '`oracle` each case’s own corpus; filters None.')
    assert 'not measured' not in line


def test_the_pool_line_omits_a_corpus_version_the_pool_does_not_have():
    pool = {'pool_depth': 10, 'settings_hash': 'abc', 'filters': None,
            'runs_corpora': {'fanout': ['ip-v1', 'criminal-v1']}}
    summary = summarize_runs(RUNS, {('c1', 'k1'): 2, ('c1', 'k2'): 0},
                             {('c1', 'k1'): 'd1', ('c1', 'k2'): 'd2'})
    line = next(line for line in render_markdown(
        summary, judge=JUDGE, pool=pool).splitlines() if line.startswith('- Pool:'))
    assert line == '- Pool: pool_depth 10, settings_hash `abc`'

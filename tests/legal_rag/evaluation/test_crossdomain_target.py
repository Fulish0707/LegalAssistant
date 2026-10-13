import pytest

from legal_rag.evaluation.diagnosis.crossdomain_target import (
    corpus_document_ids, drop_ambiguous_pairs, sample_cases, sidecar_rows, target_rows)


def rows(*specs):
    return [dict(case_id=f'qa:{i}', question=question, document_id=document_id, legal_domain='criminal',
                 answer='답변') for i, (question, document_id) in enumerate(specs)]


def test_drop_ambiguous_pairs_removes_every_row_of_a_repeated_question_document_pair():
    # load_diagnostic_rows(dataset=...) selects by (normalized question, document_id), so a pair
    # held by two rows would reproduce two case ids for one target line.
    source = rows(('질문 A', 'd1'), ('질문  A ', 'd1'), ('질문 B', 'd1'), ('질문 A', 'd2'))
    kept = drop_ambiguous_pairs(source)
    assert [row['case_id'] for row in kept] == ['qa:2', 'qa:3']


def test_sample_cases_is_seeded_and_only_draws_eligible_rows():
    source = rows(*((f'질문 {i}', f'd{i}') for i in range(20)))
    eligible = {f'd{i}' for i in range(0, 20, 2)}
    drawn = sample_cases(source, limit=4, seed=42, has_chunks=lambda doc: doc in eligible)
    assert len(drawn) == 4
    assert all(case['document_id'] in eligible for case in drawn)
    assert [c['case_id'] for c in drawn] == [c['case_id'] for c in
                                             sample_cases(source, limit=4, seed=42, has_chunks=lambda d: d in eligible)]
    assert [c['case_id'] for c in drawn] != [c['case_id'] for c in
                                             sample_cases(source, limit=4, seed=7, has_chunks=lambda d: d in eligible)]
    assert [c['case_id'] for c in drawn] == sorted(c['case_id'] for c in drawn)


def test_sample_cases_refuses_when_too_few_rows_have_chunks():
    with pytest.raises(ValueError, match='only 1'):
        sample_cases(rows(('질문 0', 'd0'), ('질문 1', 'd1')), limit=2, seed=1,
                     has_chunks=lambda doc: doc == 'd0')


def test_target_and_sidecar_rows_carry_the_case_id_and_its_corpus():
    drawn = rows(('질문 0', 'd0'))
    assert target_rows(drawn) == [{'id': 'qa:0', 'query': '질문 0', 'relevant_document_ids': ['d0']}]
    assert sidecar_rows(drawn, {'criminal': 'criminal-v1'}) == [
        {'id': 'qa:0', 'legal_domain': 'criminal', 'corpus_version': 'criminal-v1', 'document_id': 'd0'}]


def test_corpus_document_ids_reads_ids_without_validating_chunks(tmp_path):
    (tmp_path / 'documents.jsonl').write_text('{}\n', encoding='utf-8')
    (tmp_path / 'chunks.jsonl').write_text(
        '{"id": "d1#0", "document_id": "d1", "junk": 1}\n\n{"id": "d2#0", "document_id": "d2"}\n',
        encoding='utf-8')
    assert corpus_document_ids(tmp_path) == {'d1', 'd2'}


def test_corpus_document_ids_refuses_an_empty_corpus(tmp_path):
    (tmp_path / 'documents.jsonl').write_text('{}\n', encoding='utf-8')
    (tmp_path / 'chunks.jsonl').write_text('', encoding='utf-8')
    with pytest.raises(ValueError, match='no processed chunks'):
        corpus_document_ids(tmp_path)


def _script(name):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[3] / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'script_{name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('name', ['build_crossdomain_target', 'build_fanout_pool', 'summarize_fanout_runs'])
def test_script_help_exits_zero(name):
    from typer.testing import CliRunner
    assert CliRunner().invoke(_script(name).app, ['--help']).exit_code == 0


def test_the_fanout_script_refuses_a_malformed_option():
    from typer.testing import CliRunner
    script = _script('build_fanout_pool')
    result = CliRunner().invoke(script.app, ['--case-source', 'criminal=criminal-v1', '--run', 'fanout=ip-v1',
                                             '--output', 'out', '--qa-root', '.'])
    assert result.exit_code != 0

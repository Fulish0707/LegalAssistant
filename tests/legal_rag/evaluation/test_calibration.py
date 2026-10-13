import csv
import json

import pytest

from legal_rag.evaluation.relevance.calibration import (
    STRATA, agreement, read_reference_grades, sample_calibration, write_calibration)


def pool(n_cases=30):
    rows = []
    for case in range(n_cases):
        for rank in range(1, 21):
            rows.append(dict(case_id=f'c{case}', chunk_id=f'c{case}-k{rank}', document_id=f'd{rank}',
                             document_type='judgment', title='t', text='본문', question='질문',
                             reference_answer='답', case_number=f'2020허{rank}', authority='특허법원',
                             search_query=f'특허법원 2020허{rank}', date='2020.01.01',
                             ranks={'dense': rank}, is_gold_chunk=False))
        rows.append(dict(rows[-1], chunk_id=f'c{case}-gold', ranks={}, is_gold_chunk=True))
    return rows


def test_sample_meets_quotas_and_is_deterministic():
    sample = sample_calibration(pool(), seed=42)
    counts = {name: sum(row['stratum'] == name for row in sample) for name in STRATA}
    assert counts == {'gold': 10, 'rank_1_3': 15, 'rank_4_10': 15, 'rank_11_20': 10}
    assert sample == sample_calibration(list(reversed(pool())), seed=42)
    assert len({(row['case_id'], row['chunk_id']) for row in sample}) == 50
    assert len({row['case_id'] for row in sample}) > 1


def test_retrieved_gold_chunk_counts_as_gold_and_short_stratum_raises():
    rows = pool(n_cases=1)
    rows[0] = dict(rows[0], is_gold_chunk=True)
    sample = sample_calibration(rows, seed=1, quotas={'gold': 2})
    assert {row['chunk_id'] for row in sample} == {'c0-k1', 'c0-gold'}
    with pytest.raises(ValueError, match='gold'):
        sample_calibration(rows, seed=1, quotas={'gold': 3})


def test_csv_leads_with_searchable_case_number_and_hides_strata(tmp_path):
    sample = sample_calibration(pool(), seed=42)
    write_calibration(tmp_path, sample)
    with (tmp_path / 'calibration.csv').open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames[:4] == ['pair_id', 'case_number', 'search_query', 'authority']
    assert reader.fieldnames[-2:] == ['codex_grade', 'codex_note']
    assert not {'stratum', 'ranks', 'is_gold_chunk'} & set(reader.fieldnames)
    assert rows[0]['codex_grade'] == ''
    manifest = json.loads((tmp_path / 'calibration.json').read_text())
    assert len(manifest['pairs']) == 50
    with pytest.raises(FileExistsError):
        write_calibration(tmp_path, sample)


def test_write_refuses_before_creating_either_file(tmp_path):
    (tmp_path / 'calibration.csv').write_text('existing')
    with pytest.raises(FileExistsError):
        write_calibration(tmp_path, sample_calibration(pool(), seed=42))
    assert not (tmp_path / 'calibration.json').exists()
    assert (tmp_path / 'calibration.csv').read_text() == 'existing'


def test_reference_grades_reject_values_outside_0_1_2(tmp_path):
    path = tmp_path / 'calibration.csv'
    path.write_text('pair_id,codex_grade\nc1|k1,2\nc1|k2,\nc1|k3,5\n', encoding='utf-8-sig')
    with pytest.raises(ValueError, match='c1\\|k3'):
        read_reference_grades(path)
    path.write_text('pair_id,codex_grade\nc1|k1,2\nc1|k2,\n', encoding='utf-8-sig')
    assert read_reference_grades(path) == {('c1', 'k1'): 2}


def test_reference_grades_non_utf8_file_explains_how_to_resave(tmp_path):
    path = tmp_path / 'calibration.csv'
    path.write_bytes('pair_id,codex_grade,codex_note\nc1|k1,2,관련 없음\n'.encode('cp949'))
    with pytest.raises(ValueError, match='CSV UTF-8'):
        read_reference_grades(path)


def test_agreement_on_known_matrix():
    human = {('c', str(i)): g for i, g in enumerate([2, 2, 1, 0, 0, 1])}
    llm = {('c', str(i)): g for i, g in enumerate([2, 1, 1, 0, 0, 2])}
    result = agreement(human, llm, gold_pairs={('c', '0'), ('c', '3')})
    assert result['pairs'] == 6
    assert result['confusion'] == [[2, 0, 0], [0, 1, 1], [0, 1, 1]]   # rows human 0/1/2, cols llm 0/1/2
    assert result['exact_agreement'] == pytest.approx(4 / 6)
    assert result['binary_grade2']['precision'] == pytest.approx(1 / 2)
    assert result['binary_grade2']['recall'] == pytest.approx(1 / 2)
    assert result['binary_grade2']['kappa'] == pytest.approx(0.25)
    assert result['gold_llm_grades'] == {'0': 1, '1': 0, '2': 1}
    assert result['missing_llm'] == 0


def _graded_calibration(tmp_path, grades):
    from legal_rag.evaluation.relevance.calibration import CSV_FIELDS
    write_calibration(tmp_path, sample_calibration(pool(), seed=42))
    path = tmp_path / 'calibration.csv'
    with path.open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    for row, grade in zip(rows, grades):
        row['codex_grade'] = grade
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def test_require_reference_grades_refuses_missing_or_partial_labels(tmp_path):
    from legal_rag.evaluation.relevance.calibration import require_reference_grades
    with pytest.raises(ValueError, match='missing'):
        require_reference_grades(tmp_path)
    _graded_calibration(tmp_path, ['1'] * 49)
    with pytest.raises(ValueError, match='1 of 50'):
        require_reference_grades(tmp_path)


def test_require_reference_grades_accepts_complete_labels(tmp_path):
    from legal_rag.evaluation.relevance.calibration import require_reference_grades
    _graded_calibration(tmp_path, ['0'] * 50)
    assert require_reference_grades(tmp_path) == 50


def test_disagreement_rows_and_review_csv(tmp_path):
    from legal_rag.evaluation.relevance.calibration import REVIEW_FIELDS, disagreement_rows, write_review
    rows = _graded_calibration(tmp_path, ['2', '1', '0', ''])
    key = lambda row: tuple(row['pair_id'].split('|', 1))
    details = {'model_grade': 2, 'evidence_quote': '지어낸 인용', 'quote_verified': False}
    llm = {key(rows[0]): {'grade': 2, 'reason': 'same'},
           key(rows[1]): {'grade': 0, 'reason': 'differs', 'details': details},
           key(rows[2]): {'grade': None, 'reason': 'parse failure'},
           key(rows[3]): {'grade': 1, 'reason': 'no human grade'}}
    review = disagreement_rows(tmp_path / 'calibration.csv', llm)
    assert [(r['pair_id'], r['codex_grade'], r['llm_grade'], r['llm_reason']) for r in review] == [
        (rows[1]['pair_id'], 1, 0, 'differs'), (rows[2]['pair_id'], 0, None, 'parse failure')]
    write_review(tmp_path / 'review.csv', review)
    with (tmp_path / 'review.csv').open(newline='', encoding='utf-8-sig') as handle:
        written = list(csv.DictReader(handle))
    assert list(written[0]) == REVIEW_FIELDS and written[0]['case_number'] == rows[1]['case_number']
    assert {'evidence_quote', 'quote_verified', 'model_grade'} <= set(REVIEW_FIELDS)
    assert (written[0]['evidence_quote'], written[0]['quote_verified'], written[0]['model_grade']) == (
        '지어낸 인용', 'False', '2')
    assert (written[1]['evidence_quote'], written[1]['quote_verified'], written[1]['model_grade']) == ('', '', '')


def test_judge_passes_threshold():
    from legal_rag.evaluation.relevance.calibration import judge_passes
    assert judge_passes({'binary_grade2': {'kappa': 0.6}, 'missing_llm': 0})
    assert not judge_passes({'binary_grade2': {'kappa': 0.59}, 'missing_llm': 0})
    assert not judge_passes({'binary_grade2': {'kappa': None}, 'missing_llm': 0})


def test_judge_fails_when_a_reference_graded_pair_has_no_llm_grade():
    from legal_rag.evaluation.relevance.calibration import agreement, judge_passes, missing_llm_message
    human = {('c', str(i)): g for i, g in enumerate([2, 2, 0, 0, 1])}
    llm = {('c', str(i)): g for i, g in enumerate([2, 2, 0, 0])}      # pair 4 failed or not judged
    result = agreement(human, llm, gold_pairs=set())
    assert result['binary_grade2']['kappa'] == 1.0 and result['missing_llm'] == 1
    assert not judge_passes(result)
    assert '1 calibration pairs have no judge grade' in missing_llm_message(result)
    assert '--retry-failed' in missing_llm_message(result)
    assert judge_passes(dict(result, missing_llm=0))


def test_calibration_fingerprint_hashes_labels_manifest_and_pool(tmp_path):
    import hashlib
    from legal_rag.evaluation.relevance.calibration import calibration_fingerprint
    _graded_calibration(tmp_path, ['0'] * 50)
    (tmp_path / 'pool.jsonl').write_text('{}\n', encoding='utf-8')
    fingerprint = calibration_fingerprint(tmp_path)
    sha = lambda name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    assert fingerprint == {'calibration_csv_sha256': sha('calibration.csv'),
                           'calibration_json_sha256': sha('calibration.json'),
                           'pool_jsonl_sha256': sha('pool.jsonl')}
    (tmp_path / 'pool.jsonl').unlink()
    assert calibration_fingerprint(tmp_path)['pool_jsonl_sha256'] is None


def test_model_grades_fall_back_to_stored_grade_and_downgrades_are_counted():
    from legal_rag.evaluation.relevance.calibration import model_grades, quote_downgrades
    rows = {('c', '1'): {'grade': 1, 'details': {'model_grade': 2, 'evidence_quote': 'x', 'quote_verified': False}},
            ('c', '2'): {'grade': 2, 'details': {'model_grade': 2, 'evidence_quote': 'y', 'quote_verified': True}},
            ('c', '3'): {'grade': 0},                                   # v1 row without details
            ('c', '4'): {'grade': None, 'details': None},               # parse failure
            ('c', '5'): {'grade': 1, 'details': {'model_grade': 2, 'evidence_quote': '', 'quote_verified': False}}}
    assert model_grades(rows) == {('c', '1'): 2, ('c', '2'): 2, ('c', '3'): 0, ('c', '5'): 2}
    assert quote_downgrades(rows, {('c', '1'), ('c', '2'), ('c', '3'), ('c', '4')}) == 1


def _load_script(name):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[3] / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'script_{name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_agreement_script_writes_per_key_files_with_reference_source(tmp_path):
    from typer.testing import CliRunner
    from legal_rag.evaluation.relevance.judgment_store import JudgmentStore
    from legal_rag.evaluation.relevance.relevance_judge import judge_key
    rows = _graded_calibration(tmp_path, ['2'] * 25 + ['0'] * 25)
    (tmp_path / 'pool.jsonl').write_text(
        ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in pool()), encoding='utf-8')
    (tmp_path / 'agreement.json').write_text('v1', encoding='utf-8')
    (tmp_path / 'calibration_review.csv').write_text('v1', encoding='utf-8')
    key = judge_key('ollama:qwen3:8b', think=True)
    store = JudgmentStore(tmp_path / 'judgments.jsonl')
    for number, row in enumerate(rows):
        case_id, chunk_id = row['pair_id'].split('|', 1)
        human = int(row['codex_grade'])
        downgraded = number == 0
        store.append(key, case_id=case_id, chunk_id=chunk_id, document_id='d', reason='r',
                     grade=1 if downgraded else human,
                     details={'model_grade': human, 'evidence_quote': 'q', 'quote_verified': not downgraded})
    script = _load_script('judge_agreement')
    result = CliRunner().invoke(script.app, ['--pool-dir', str(tmp_path), '--think',
                                             '--reference-source', 'GPT 6 Astra'])
    assert 'reference labels: GPT 6 Astra' in result.output
    assert key.prompt_sha256[:12] in result.output
    report = json.loads((tmp_path / f'agreement-{key.prompt_sha256[:12]}.json').read_text(encoding='utf-8'))
    assert report['reference_source'] == 'GPT 6 Astra' and report['think'] is True
    assert report['prompt_sha256'] == key.prompt_sha256
    assert report['quote_downgrades'] == 1
    assert report['exact_agreement'] == pytest.approx(49 / 50)
    assert report['agreement_before_quote_check']['exact_agreement'] == 1.0
    assert (tmp_path / f'calibration_review-{key.prompt_sha256[:12]}.csv').exists()
    assert (tmp_path / 'agreement.json').read_text(encoding='utf-8') == 'v1'
    assert (tmp_path / 'calibration_review.csv').read_text(encoding='utf-8') == 'v1'
    default = CliRunner().invoke(script.app, ['--pool-dir', str(tmp_path)])
    assert 'reference labels: codex' in default.output and default.exit_code == 1   # no think-off judgments


def test_scripts_help_exits_zero():
    from typer.testing import CliRunner
    for name in ('judge_relevance', 'judge_agreement', 'eval_graded'):
        result = CliRunner().invoke(_load_script(name).app, ['--help'])
        assert result.exit_code == 0, (name, result.output)
        assert '--think' in result.output and '--no-think' in result.output


def test_sample_excludes_given_pair_ids():
    rows = pool(n_cases=2)
    top3 = [f"{r['case_id']}|{r['chunk_id']}" for r in rows if r['ranks'] and r['ranks']['dense'] <= 3]
    exclude = set(top3[:4])
    sample = sample_calibration(rows, seed=43, quotas={'rank_1_3': 2}, exclude=exclude)
    assert {f"{r['case_id']}|{r['chunk_id']}" for r in sample} == set(top3[4:])
    with pytest.raises(ValueError, match='rank_1_3 has 2 pairs'):
        sample_calibration(rows, seed=43, quotas={'rank_1_3': 3}, exclude=exclude)


def test_without_a_gold_quota_retrieved_gold_chunks_fall_into_rank_strata():
    rows = [dict(row, is_gold_chunk=True) if row['chunk_id'] == 'c0-k1' else row for row in pool(n_cases=1)]
    sample = sample_calibration(rows, seed=1, quotas={'rank_1_3': 3, 'rank_4_10': 1})
    assert {'c0-k1', 'c0-k2', 'c0-k3'} <= {row['chunk_id'] for row in sample}
    assert 'c0-gold' not in {row['chunk_id'] for row in sample}          # unretrieved gold has no rank
    with pytest.raises(ValueError, match='unknown stratum'):
        sample_calibration(rows, seed=1, quotas={'rank_1_5': 1})


def test_parse_quotas_keeps_argument_order_and_validates():
    from legal_rag.evaluation.relevance.calibration import parse_quotas
    assert list(parse_quotas(['rank_4_10=25', 'rank_1_3=25']).items()) == [('rank_4_10', 25), ('rank_1_3', 25)]
    assert parse_quotas([]) == STRATA
    for bad in (['rank_1_3'], ['rank_1_3=x'], ['nope=3'], ['rank_1_3=1', 'rank_1_3=2'], ['rank_1_3=0']):
        with pytest.raises(ValueError):
            parse_quotas(bad)


def test_sample_script_quotas_and_exclusion(tmp_path):
    from typer.testing import CliRunner
    first, second = tmp_path / 'first', tmp_path / 'second'
    for directory in (first, second):
        directory.mkdir()
        (directory / 'pool.jsonl').write_text(
            ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in pool(n_cases=3)), encoding='utf-8')
    write_calibration(first, sample_calibration(pool(n_cases=3), seed=42, quotas={'rank_1_3': 8}))
    first_ids = {p['pair_id'] for p in json.loads((first / 'calibration.json').read_text())['pairs']}
    script = _load_script('sample_calibration')
    result = CliRunner().invoke(script.app, ['--pool-dir', str(second), '--seed', '43', '--quota', 'rank_1_3=1',
                                             '--quota', 'rank_4_10=2', '--exclude-calibration', str(first)])
    assert result.exit_code == 0, result.output
    pairs = json.loads((second / 'calibration.json').read_text())['pairs']
    assert sorted(p['stratum'] for p in pairs) == ['rank_1_3', 'rank_4_10', 'rank_4_10']
    assert not first_ids & {p['pair_id'] for p in pairs}
    assert 'excluded 8' in result.output
    bad = CliRunner().invoke(script.app, ['--pool-dir', str(first), '--quota', 'bogus=1'])
    assert bad.exit_code == 1


def test_scripts_without_think_help_exits_zero():
    from typer.testing import CliRunner
    result = CliRunner().invoke(_load_script('sample_calibration').app, ['--help'])
    assert result.exit_code == 0 and '--quota' in result.output and '--exclude-calibration' in result.output


def _matrix_pairs(matrix):
    human, llm, number = {}, {}, 0
    for ref, row in enumerate(matrix):
        for judge, count in enumerate(row):
            for _ in range(count):
                human[('c', str(number))], llm[('c', str(number))] = ref, judge
                number += 1
    return human, llm


def test_agreement_reports_both_binary_boundaries_on_calibration_matrix():
    from legal_rag.evaluation.relevance.calibration import judge_passes
    human, llm = _matrix_pairs([[16, 4, 1], [3, 6, 8], [0, 4, 8]])
    result = agreement(human, llm, gold_pairs=set())
    assert result['confusion'] == [[16, 4, 1], [3, 6, 8], [0, 4, 8]]
    assert result['binary_relevant']['kappa'] == pytest.approx(0.6672, abs=1e-4)
    assert result['binary_relevant']['precision'] == pytest.approx(26 / 31)
    assert result['binary_relevant']['recall'] == pytest.approx(26 / 29)
    assert result['binary_grade2']['kappa'] == pytest.approx(0.376, abs=1e-3)
    assert judge_passes(result, 1) and not judge_passes(result, 2) and not judge_passes(result)
    assert not judge_passes(dict(result, missing_llm=1), 1)
    assert not judge_passes(result, 1, threshold=0.7)
    with pytest.raises(ValueError):
        judge_passes(result, 0)


def test_agreement_script_pass_boundary(tmp_path):
    from typer.testing import CliRunner
    from legal_rag.evaluation.relevance.judgment_store import JudgmentStore
    from legal_rag.evaluation.relevance.relevance_judge import judge_key
    rows = _graded_calibration(tmp_path, ['2'] * 20 + ['1'] * 10 + ['0'] * 20)
    (tmp_path / 'pool.jsonl').write_text(
        ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in pool()), encoding='utf-8')
    key = judge_key('ollama:qwen3:8b', think=True)
    store = JudgmentStore(tmp_path / 'judgments.jsonl')
    for row in rows:                                   # judge swaps 1 and 2: >=1 perfect, grade 2 poor
        case_id, chunk_id = row['pair_id'].split('|', 1)
        grade = {'2': 1, '1': 2, '0': 0}[row['codex_grade']]
        store.append(key, case_id=case_id, chunk_id=chunk_id, document_id='d', reason='r', grade=grade)
    script = _load_script('judge_agreement')
    passed = CliRunner().invoke(script.app, ['--pool-dir', str(tmp_path), '--think', '--pass-boundary', '1'])
    assert passed.exit_code == 0, passed.output
    assert 'PASS: binary grade≥1 kappa 1.000' in passed.output
    report = json.loads((tmp_path / f'agreement-{key.prompt_sha256[:12]}.json').read_text(encoding='utf-8'))
    assert report['pass_boundary'] == 1 and report['binary_relevant']['kappa'] == 1.0
    assert 'binary_relevant' in report['agreement_before_quote_check']
    failed = CliRunner().invoke(script.app, ['--pool-dir', str(tmp_path), '--think'])
    assert failed.exit_code == 1 and 'FAIL: binary grade≥2 kappa' in failed.output
    assert json.loads((tmp_path / f'agreement-{key.prompt_sha256[:12]}.json').read_text())['pass_boundary'] == 2
    assert CliRunner().invoke(script.app, ['--pool-dir', str(tmp_path), '--think', '--pass-boundary', '3']).exit_code != 0


def test_reference_grades_fall_back_to_legacy_human_columns(tmp_path):
    path = tmp_path / 'calibration.csv'
    path.write_text('pair_id,human_grade,human_note\nc1|k1,1,old\n', encoding='utf-8-sig')
    assert read_reference_grades(path) == {('c1', 'k1'): 1}


def test_codex_columns_win_over_empty_legacy_columns(tmp_path):
    # A file whose labeller added codex_* next to the original, still empty human_* columns.
    path = tmp_path / 'calibration.csv'
    path.write_text('pair_id,human_grade,human_note,codex_grade,codex_note\nc1|k1,,,2,근거\nc1|k2,,,,\n',
                    encoding='utf-8-sig')
    assert read_reference_grades(path) == {('c1', 'k1'): 2}


def test_conflicting_codex_and_legacy_grades_are_refused(tmp_path):
    path = tmp_path / 'calibration.csv'
    path.write_text('pair_id,human_grade,codex_grade\nc1|k1,1,2\n', encoding='utf-8-sig')
    with pytest.raises(ValueError, match='conflicting'):
        read_reference_grades(path)


def test_deep_ranks_form_their_own_stratum():
    from legal_rag.evaluation.relevance.calibration import _stratum

    assert _stratum({'is_gold_chunk': False, 'ranks': {'hybrid': 15}}) == 'rank_11_20'
    assert _stratum({'is_gold_chunk': False, 'ranks': {'hybrid': 21}}) == 'rank_21_50'
    assert _stratum({'is_gold_chunk': False, 'ranks': {'hybrid': 50}}) == 'rank_21_50'
    assert _stratum({'is_gold_chunk': False, 'ranks': {'hybrid': 51}}) is None


class TestPrevalenceRobustAgreement:
    def test_ac1_matches_a_hand_computed_value(self):
        from legal_rag.evaluation.relevance.calibration import agreement

        # 50 pairs: reference calls 40 relevant, judge 35, 33 of them shared, 8 shared zeros.
        reference, llm, number = {}, {}, 0
        for ref, judge, count in ((1, 1, 33), (1, 0, 7), (0, 1, 2), (0, 0, 8)):
            for _ in range(count):
                reference[('c', str(number))], llm[('c', str(number))] = ref, judge
                number += 1
        result = agreement(reference, llm, gold_pairs=set())['binary_relevant']
        assert result['accuracy'] == pytest.approx(0.82)
        assert result['prevalence'] == pytest.approx(0.80)
        assert result['kappa'] == pytest.approx(0.526, abs=5e-4)      # dragged down by prevalence
        assert result['ac1'] == pytest.approx(0.712, abs=5e-4)

    def test_ac1_holds_where_kappa_collapses_on_the_same_accuracy(self):
        from legal_rag.evaluation.relevance.calibration import _gwet_ac1, _kappa

        balanced = ([1] * 25 + [0] * 25, [1] * 21 + [0] * 4 + [1] * 5 + [0] * 20)
        skewed = ([1] * 45 + [0] * 5, [1] * 41 + [0] * 4 + [1] * 5 + [0] * 0)
        for reference, judge in (balanced, skewed):
            assert sum(x == y for x, y in zip(reference, judge)) / 50 == pytest.approx(0.82)
        assert _kappa(*balanced, weighted=False) > 0.6 > _kappa(*skewed, weighted=False)
        assert _gwet_ac1(*balanced) > 0.6 and _gwet_ac1(*skewed) > 0.6

    def test_an_unknown_metric_is_refused(self):
        from legal_rag.evaluation.relevance.calibration import judge_passes

        with pytest.raises(ValueError, match='kappa'):
            judge_passes({'missing_llm': 0, 'binary_relevant': {'kappa': 1.0}}, 1, metric='f1')

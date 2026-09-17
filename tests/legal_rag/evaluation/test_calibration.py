import csv
import json

import pytest

from legal_rag.evaluation.calibration import (
    STRATA, agreement, read_human_grades, sample_calibration, write_calibration)


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
    assert reader.fieldnames[-2:] == ['human_grade', 'human_note']
    assert not {'stratum', 'ranks', 'is_gold_chunk'} & set(reader.fieldnames)
    assert rows[0]['human_grade'] == ''
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


def test_human_grades_reject_values_outside_0_1_2(tmp_path):
    path = tmp_path / 'calibration.csv'
    path.write_text('pair_id,human_grade\nc1|k1,2\nc1|k2,\nc1|k3,5\n', encoding='utf-8-sig')
    with pytest.raises(ValueError, match='c1\\|k3'):
        read_human_grades(path)
    path.write_text('pair_id,human_grade\nc1|k1,2\nc1|k2,\n', encoding='utf-8-sig')
    assert read_human_grades(path) == {('c1', 'k1'): 2}


def test_human_grades_non_utf8_file_explains_how_to_resave(tmp_path):
    path = tmp_path / 'calibration.csv'
    path.write_bytes('pair_id,human_grade,human_note\nc1|k1,2,관련 없음\n'.encode('cp949'))
    with pytest.raises(ValueError, match='CSV UTF-8'):
        read_human_grades(path)


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
    from legal_rag.evaluation.calibration import CSV_FIELDS
    write_calibration(tmp_path, sample_calibration(pool(), seed=42))
    path = tmp_path / 'calibration.csv'
    with path.open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    for row, grade in zip(rows, grades):
        row['human_grade'] = grade
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def test_require_human_grades_refuses_missing_or_partial_labels(tmp_path):
    from legal_rag.evaluation.calibration import require_human_grades
    with pytest.raises(ValueError, match='missing'):
        require_human_grades(tmp_path)
    _graded_calibration(tmp_path, ['1'] * 49)
    with pytest.raises(ValueError, match='1 of 50'):
        require_human_grades(tmp_path)


def test_require_human_grades_accepts_complete_labels(tmp_path):
    from legal_rag.evaluation.calibration import require_human_grades
    _graded_calibration(tmp_path, ['0'] * 50)
    assert require_human_grades(tmp_path) == 50


def test_disagreement_rows_and_review_csv(tmp_path):
    from legal_rag.evaluation.calibration import REVIEW_FIELDS, disagreement_rows, write_review
    rows = _graded_calibration(tmp_path, ['2', '1', '0', ''])
    key = lambda row: tuple(row['pair_id'].split('|', 1))
    llm = {key(rows[0]): {'grade': 2, 'reason': 'same'},
           key(rows[1]): {'grade': 0, 'reason': 'differs'},
           key(rows[2]): {'grade': None, 'reason': 'parse failure'},
           key(rows[3]): {'grade': 1, 'reason': 'no human grade'}}
    review = disagreement_rows(tmp_path / 'calibration.csv', llm)
    assert [(r['pair_id'], r['human_grade'], r['llm_grade'], r['llm_reason']) for r in review] == [
        (rows[1]['pair_id'], 1, 0, 'differs'), (rows[2]['pair_id'], 0, None, 'parse failure')]
    write_review(tmp_path / 'review.csv', review)
    with (tmp_path / 'review.csv').open(newline='', encoding='utf-8-sig') as handle:
        written = list(csv.DictReader(handle))
    assert list(written[0]) == REVIEW_FIELDS and written[0]['case_number'] == rows[1]['case_number']


def test_judge_passes_threshold():
    from legal_rag.evaluation.calibration import judge_passes
    assert judge_passes({'binary_grade2': {'kappa': 0.6}, 'missing_llm': 0})
    assert not judge_passes({'binary_grade2': {'kappa': 0.59}, 'missing_llm': 0})
    assert not judge_passes({'binary_grade2': {'kappa': None}, 'missing_llm': 0})


def test_judge_fails_when_a_human_graded_pair_has_no_llm_grade():
    from legal_rag.evaluation.calibration import agreement, judge_passes, missing_llm_message
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
    from legal_rag.evaluation.calibration import calibration_fingerprint
    _graded_calibration(tmp_path, ['0'] * 50)
    (tmp_path / 'pool.jsonl').write_text('{}\n', encoding='utf-8')
    fingerprint = calibration_fingerprint(tmp_path)
    sha = lambda name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    assert fingerprint == {'calibration_csv_sha256': sha('calibration.csv'),
                           'calibration_json_sha256': sha('calibration.json'),
                           'pool_jsonl_sha256': sha('pool.jsonl')}
    (tmp_path / 'pool.jsonl').unlink()
    assert calibration_fingerprint(tmp_path)['pool_jsonl_sha256'] is None

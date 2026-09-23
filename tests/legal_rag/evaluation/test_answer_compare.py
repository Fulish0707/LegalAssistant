import pytest

from legal_rag.evaluation.answer_compare import CSV_FIELDS, pairwise_rows, sample_cases, win_rate


def answers(count=4):
    return [{'case_id': f'c{i}', 'question': f'질문 {i}', 'ip-v1': f'v1 답변 {i}', 'ip-v2': f'v2 답변 {i}'}
            for i in range(count)]


def test_sides_are_randomised_but_reproducible():
    rows, key = pairwise_rows(answers(), seed=42)
    again, key_again = pairwise_rows(answers(), seed=42)
    assert [row['case_id'] for row in rows] == ['c0', 'c1', 'c2', 'c3']
    assert (rows, key) == (again, key_again)
    assert set(key.values()) == {'A=ip-v1,B=ip-v2', 'A=ip-v2,B=ip-v1'}, 'both orders must occur'
    for row in rows:
        assert set(row) == set(CSV_FIELDS) and row['winner'] == ''


def test_the_csv_never_names_the_corpus():
    rows, _ = pairwise_rows(answers(), seed=7)
    text = ' '.join(value for row in rows for value in row.values())
    assert 'ip-v1' not in text and 'ip-v2' not in text


def test_win_rate_reads_the_hidden_mapping():
    key = {'c0': 'A=ip-v1,B=ip-v2', 'c1': 'A=ip-v2,B=ip-v1', 'c2': 'A=ip-v1,B=ip-v2'}
    verdicts = {'c0': 'B', 'c1': 'A', 'c2': 'tie'}          # ip-v2 wins twice, one tie
    result = win_rate(verdicts, key, side='ip-v2')
    assert (result['wins'], result['losses'], result['ties']) == (2, 0, 1)
    assert result['win_rate'] == pytest.approx(5 / 6)


def test_a_side_that_always_loses_is_decisive():
    key = {f'c{i}': 'A=ip-v1,B=ip-v2' for i in range(30)}
    result = win_rate({case: 'A' for case in key}, key, side='ip-v2')
    assert result['win_rate'] == 0.0 and result['decisive']


def test_an_even_split_is_not_decisive():
    key = {f'c{i}': 'A=ip-v1,B=ip-v2' for i in range(30)}
    verdicts = {case: ('A' if index % 2 else 'B') for index, case in enumerate(key)}
    assert not win_rate(verdicts, key, side='ip-v2')['decisive']


def test_an_unknown_verdict_is_refused():
    with pytest.raises(ValueError, match='winner'):
        win_rate({'c0': 'both'}, {'c0': 'A=ip-v1,B=ip-v2'}, side='ip-v2')


def test_sampling_is_deterministic_and_bounded():
    assert sample_cases([f'c{i}' for i in range(10)], limit=3, seed=1) == \
           sample_cases([f'c{i}' for i in range(10)], limit=3, seed=1)
    assert len(sample_cases([f'c{i}' for i in range(10)], limit=3, seed=1)) == 3
    assert len(sample_cases(['a', 'b'], limit=5, seed=1)) == 2


def test_a_case_keeps_its_sides_when_the_sample_changes():
    small, key_small = pairwise_rows(answers(2), seed=42)
    large, key_large = pairwise_rows(answers(6), seed=42)
    assert key_small['c1'] == key_large['c1']
    by_case = {row['case_id']: row for row in large}
    assert by_case['c1']['answer_a'] == next(r for r in small if r['case_id'] == 'c1')['answer_a']

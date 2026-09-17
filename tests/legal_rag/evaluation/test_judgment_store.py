import json

import pytest

from legal_rag.evaluation.judgment_store import JudgeKey, JudgmentStore

KEY = JudgeKey(model='ollama:qwen3:8b', prompt_sha256='p1')


def test_append_then_reload(tmp_path):
    store = JudgmentStore(tmp_path / 'judgments.jsonl')
    store.append(KEY, case_id='c1', chunk_id='k1', document_id='d1', grade=2, reason='r')
    reloaded = JudgmentStore(tmp_path / 'judgments.jsonl')
    assert reloaded.grades(KEY) == {('c1', 'k1'): 2}
    assert reloaded.has(KEY, 'c1', 'k1')
    assert reloaded.documents(KEY) == {('c1', 'k1'): 'd1'}


def test_other_model_or_prompt_is_a_separate_judgment(tmp_path):
    store = JudgmentStore(tmp_path / 'judgments.jsonl')
    store.append(KEY, case_id='c1', chunk_id='k1', document_id='d1', grade=2, reason='')
    other = JudgeKey(model='ollama:qwen3:8b', prompt_sha256='p2')
    assert not store.has(other, 'c1', 'k1')
    store.append(other, case_id='c1', chunk_id='k1', document_id='d1', grade=0, reason='')
    assert store.grades(KEY) == {('c1', 'k1'): 2}
    assert store.grades(other) == {('c1', 'k1'): 0}


def test_null_grade_is_recorded_but_not_a_grade(tmp_path):
    store = JudgmentStore(tmp_path / 'judgments.jsonl')
    store.append(KEY, case_id='c1', chunk_id='k1', document_id='d1', grade=None, reason='', raw='garbage')
    assert store.has(KEY, 'c1', 'k1')          # never re-judged on resume
    assert store.grades(KEY) == {}
    assert store.failures(KEY) == [('c1', 'k1')]


def test_duplicate_append_is_refused(tmp_path):
    store = JudgmentStore(tmp_path / 'judgments.jsonl')
    store.append(KEY, case_id='c1', chunk_id='k1', document_id='d1', grade=1, reason='')
    with pytest.raises(ValueError, match='already judged'):
        store.append(KEY, case_id='c1', chunk_id='k1', document_id='d1', grade=2, reason='')


def test_truncated_final_line_is_dropped_and_appends_stay_readable(tmp_path):
    path = tmp_path / 'judgments.jsonl'
    store = JudgmentStore(path)
    store.append(KEY, case_id='c1', chunk_id='k1', document_id='d1', grade=1, reason='')
    with path.open('a', encoding='utf-8') as handle:
        handle.write('{"case_id": "c1", "chunk_')
    resumed = JudgmentStore(path)
    assert resumed.grades(KEY) == {('c1', 'k1'): 1}
    resumed.append(KEY, case_id='c1', chunk_id='k2', document_id='d1', grade=2, reason='')
    assert JudgmentStore(path).grades(KEY) == {('c1', 'k1'): 1, ('c1', 'k2'): 2}


def test_corruption_before_the_final_line_raises(tmp_path):
    path = tmp_path / 'judgments.jsonl'
    row = {'case_id': 'c1', 'chunk_id': 'k1', 'document_id': 'd1', 'grade': 1, 'reason': '',
           'judge_model': KEY.model, 'prompt_sha256': KEY.prompt_sha256, 'judged_at': '', 'raw': None}
    path.write_text('{"broken\n' + json.dumps(row) + '\n', encoding='utf-8')
    with pytest.raises(json.JSONDecodeError):
        JudgmentStore(path)


def test_rows_returns_reason_for_one_judge_only(tmp_path):
    store = JudgmentStore(tmp_path / 'judgments.jsonl')
    store.append(KEY, case_id='c1', chunk_id='k1', document_id='d1', grade=1, reason='why')
    store.append(JudgeKey('other', 'p1'), case_id='c1', chunk_id='k1', document_id='d1', grade=2, reason='x')
    rows = JudgmentStore(tmp_path / 'judgments.jsonl').rows(KEY)
    assert list(rows) == [('c1', 'k1')]
    assert (rows[('c1', 'k1')]['grade'], rows[('c1', 'k1')]['reason']) == (1, 'why')


def test_append_over_failed_row_is_allowed_and_later_row_wins_on_reload(tmp_path):
    store = JudgmentStore(tmp_path / 'judgments.jsonl')
    store.append(KEY, case_id='c1', chunk_id='k1', document_id='d1', grade=None, reason='', raw='garbage')
    assert store.is_failed(KEY, 'c1', 'k1')
    store.append(KEY, case_id='c1', chunk_id='k1', document_id='d1', grade=2, reason='ok')
    reloaded = JudgmentStore(tmp_path / 'judgments.jsonl')
    assert reloaded.grades(KEY) == {('c1', 'k1'): 2} and reloaded.failures(KEY) == []
    assert not reloaded.is_failed(KEY, 'c1', 'k1')
    with pytest.raises(ValueError, match='already judged'):
        reloaded.append(KEY, case_id='c1', chunk_id='k1', document_id='d1', grade=None, reason='')


def test_details_are_stored_and_rows_without_details_still_load(tmp_path):
    path = tmp_path / 'judgments.jsonl'
    legacy = {'case_id': 'c0', 'chunk_id': 'k0', 'document_id': 'd0', 'grade': 1, 'reason': 'v1',
              'judge_model': KEY.model, 'prompt_sha256': KEY.prompt_sha256, 'judged_at': 'x', 'raw': None}
    path.write_text(json.dumps(legacy, ensure_ascii=False) + '\n', encoding='utf-8')
    store = JudgmentStore(path)
    details = {'model_grade': 2, 'evidence_quote': '인용', 'quote_verified': False}
    store.append(KEY, case_id='c1', chunk_id='k1', document_id='d1', grade=1, reason='r', details=details)
    store.append(KEY, case_id='c2', chunk_id='k2', document_id='d2', grade=0, reason='r')
    rows = JudgmentStore(path).rows(KEY)
    assert rows[('c0', 'k0')]['grade'] == 1 and 'details' not in rows[('c0', 'k0')]
    assert rows[('c1', 'k1')]['details'] == details
    assert rows[('c2', 'k2')]['details'] is None

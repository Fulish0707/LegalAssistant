from legal_rag.evaluation.judgment_store import JudgmentStore
import pytest

from legal_rag.evaluation.relevance_judge import (JUDGE_TEMPERATURE, RelevanceGrade, generation_settings, judge_key,
                                                  judge_pool, verify_quote)
from legal_rag.llm.base import ModelResponse, StructuredOutputError


class ScriptedModel:
    name = 'scripted'

    def __init__(self, replies, think=False):
        self.replies, self.messages, self.settings, self.think = list(replies), [], [], think

    def generate(self, messages, *, response_schema=None, temperature=0.2, max_tokens=1024):
        self.messages.append(messages)
        self.settings.append((response_schema, temperature, max_tokens))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ModelResponse(text='', parsed=RelevanceGrade(**{'evidence_quote': '', **reply}))


def pair(chunk_id='k1', **extra):
    fields = dict(case_id='c1', chunk_id=chunk_id, document_id='d1', document_type='judgment',
                  title='상표법위반', text='부정경쟁의 목적이란 …', question='부정경쟁의 목적은?',
                  reference_answer='신용을 이용하여 부당한 이익을 얻을 목적', ranks={'hybrid': 3},
                  is_gold_chunk=True)
    fields.update(extra)
    return fields


def test_grades_are_appended_and_resume_skips_judged(tmp_path):
    store = JudgmentStore(tmp_path / 'j.jsonl')
    model = ScriptedModel([{'grade': 2, 'reason': '법리 명시', 'evidence_quote': QUOTE}])
    assert judge_pool([pair(text=CHUNK)], model, store, think=False) == {'judged': 1, 'skipped': 0, 'failed': 0}
    again = judge_pool([pair()], ScriptedModel([]), store, think=False)
    assert again == {'judged': 0, 'skipped': 1, 'failed': 0}
    assert store.grades(judge_key(model, think=False)) == {('c1', 'k1'): 2}


def test_prompt_hides_rank_mode_and_gold_status(tmp_path):
    model = ScriptedModel([{'grade': 1, 'reason': ''}])
    judge_pool([pair()], model, JudgmentStore(tmp_path / 'j.jsonl'), think=False)
    prompt = ' '.join(message.content for message in model.messages[0])
    assert '부정경쟁의 목적은?' in prompt and '부당한 이익' in prompt and '상표법위반' in prompt
    for leaked in ('hybrid', 'gold', 'rank', 'is_gold_chunk'):
        assert leaked not in prompt


def test_malformed_output_retries_once_then_records_null_not_zero(tmp_path):
    store = JudgmentStore(tmp_path / 'j.jsonl')
    bad = StructuredOutputError('bad', 'not json')
    model = ScriptedModel([bad, bad])
    assert judge_pool([pair()], model, store, think=False) == {'judged': 0, 'skipped': 0, 'failed': 1}
    assert len(model.messages) == 2
    assert store.grades(judge_key(model, think=False)) == {}
    assert store.failures(judge_key(model, think=False)) == [('c1', 'k1')]


def test_retry_success_counts_as_judged(tmp_path):
    model = ScriptedModel([StructuredOutputError('bad', 'x'), {'grade': 0, 'reason': ''}])
    assert judge_pool([pair()], model, JudgmentStore(tmp_path / 'j.jsonl'), think=False)['judged'] == 1


def test_long_chunks_are_truncated_in_the_prompt(tmp_path):
    model = ScriptedModel([{'grade': 0, 'reason': ''}])
    judge_pool([pair(text='가' * 10_000)], model, JudgmentStore(tmp_path / 'j.jsonl'), think=False)
    assert '가' * 3600 not in ' '.join(m.content for m in model.messages[0])


def test_braces_in_chunk_text_and_title_are_passed_through(tmp_path):
    model = ScriptedModel([{'grade': 0, 'reason': ''}])
    judge_pool([pair(text='조건 {a} 및 {0} }{', title='{title}')], model, JudgmentStore(tmp_path / 'j.jsonl'), think=False)
    user = model.messages[0][1].content
    assert '조건 {a} 및 {0} }{' in user and '{title}' in user


def test_transport_errors_propagate_without_recording(tmp_path):
    store = JudgmentStore(tmp_path / 'j.jsonl')
    with pytest.raises(TimeoutError):
        judge_pool([pair()], ScriptedModel([TimeoutError('slow')]), store, think=False)
    assert not store.has(judge_key(ScriptedModel([]), think=False), 'c1', 'k1')


def test_generate_uses_the_hashed_generation_settings(tmp_path):
    model = ScriptedModel([{'grade': 1, 'reason': ''}])
    judge_pool([pair()], model, JudgmentStore(tmp_path / 'j.jsonl'), think=False)
    assert model.settings == [(RelevanceGrade, JUDGE_TEMPERATURE, 512)]
    thinking = ScriptedModel([{'grade': 1, 'reason': ''}], think=True)
    judge_pool([pair()], thinking, JudgmentStore(tmp_path / 'j.jsonl'), think=True)
    assert thinking.settings == [(RelevanceGrade, JUDGE_TEMPERATURE, 4096)]


def test_retry_failed_rejudges_only_failed_pairs(tmp_path):
    store = JudgmentStore(tmp_path / 'j.jsonl')
    bad = StructuredOutputError('bad', text='garbage')
    model = ScriptedModel([{'grade': 1, 'reason': 'ok'}, bad, bad])
    judge_pool([pair('k1'), pair('k2')], model, store, think=False)
    assert store.failures(judge_key(model, think=False)) == [('c1', 'k2')]
    assert judge_pool([pair('k1'), pair('k2')], ScriptedModel([]), store, think=False) == {'judged': 0, 'skipped': 2, 'failed': 0}
    retry = ScriptedModel([{'grade': 2, 'reason': 'now', 'evidence_quote': QUOTE}])
    counts = judge_pool([pair('k1'), pair('k2', text=CHUNK)], retry, store, think=False, retry_failed=True)
    assert counts == {'judged': 1, 'skipped': 1, 'failed': 0}
    reloaded = JudgmentStore(tmp_path / 'j.jsonl')
    assert reloaded.grades(judge_key(model, think=False)) == {('c1', 'k1'): 1, ('c1', 'k2'): 2}


def test_think_is_part_of_the_key_and_generation_settings():
    from legal_rag.evaluation.relevance_judge import JUDGE_THINK, prompt_sha256
    assert JUDGE_THINK is False
    assert prompt_sha256(think=False) != prompt_sha256(think=True)
    model = ScriptedModel([])
    assert judge_key(model, think=False).prompt_sha256 == prompt_sha256(think=False)
    assert judge_key(model, think=True).prompt_sha256 == prompt_sha256(think=True)
    assert generation_settings(think=False) == {'max_tokens': 512, 'num_ctx': 8192}
    assert generation_settings(think=True) == {'max_tokens': 4096, 'num_ctx': 8192}


def test_judge_pool_refuses_a_model_whose_think_mode_differs(tmp_path):
    with pytest.raises(ValueError, match='think'):
        judge_pool([pair()], ScriptedModel([], think=True), JudgmentStore(tmp_path / 'j.jsonl'), think=False)


def test_judge_run_record_and_append(tmp_path):
    import json
    from legal_rag.evaluation.judgment_store import JudgeKey
    from legal_rag.evaluation.relevance_judge import append_judge_run, judge_run_record
    record = judge_run_record(JudgeKey('ollama:qwen3:8b', 'p1'), calibration_csv_sha256='abc',
                              calibration_only=True, retry_failed=False, pairs=50, think=True,
                              counts={'judged': 49, 'skipped': 0, 'failed': 1}, model_digest=None)
    assert record['judge_model'] == 'ollama:qwen3:8b' and record['prompt_sha256'] == 'p1'
    assert record['calibration_csv_sha256'] == 'abc' and record['model_digest'] is None
    assert record['calibration_only'] is True and record['retry_failed'] is False
    assert record['think'] is True and record['generation'] == {'max_tokens': 4096, 'num_ctx': 8192}
    assert record['pairs'] == 50 and record['counts']['failed'] == 1 and record['finished_at']
    append_judge_run(tmp_path, record)
    append_judge_run(tmp_path, dict(record, retry_failed=True))
    lines = (tmp_path / 'judge_runs.jsonl').read_text(encoding='utf-8').splitlines()
    assert [json.loads(line)['retry_failed'] for line in lines] == [False, True]


def test_ollama_model_digest_is_best_effort():
    import httpx
    from legal_rag.evaluation.relevance_judge import ollama_model_digest
    tags = {'models': [{'name': 'llama3:8b', 'digest': 'x'}, {'name': 'qwen3:8b', 'digest': 'sha-q'}]}
    ok = httpx.MockTransport(lambda request: httpx.Response(200, json=tags))
    assert ollama_model_digest('http://ollama', 'qwen3:8b', transport=ok) == 'sha-q'
    assert ollama_model_digest('http://ollama', 'absent', transport=ok) is None

    def refuse(request):
        raise httpx.ConnectError('down')
    assert ollama_model_digest('http://ollama', 'qwen3:8b', transport=httpx.MockTransport(refuse)) is None
    broken = httpx.MockTransport(lambda request: httpx.Response(500))
    assert ollama_model_digest('http://ollama', 'qwen3:8b', transport=broken) is None


QUOTE = '수요자를 기만할 염려가 있는 상표를 등록받을 수 없다'
CHUNK = '상표법 제34조 제1항 제12호는 수요자를 기만할 염려가 있는 상표를 등록받을 수 없다고 규정한다. 이하 생략.'


def test_verify_quote_accepts_verbatim_and_normalized_quotes():
    assert verify_quote('수요자를 기만할 염려가 있는 상표를 등록받을 수 없다', CHUNK)
    assert verify_quote('  수요자를   기만할\n염려가 있는 상표를 등록받을 수 없다 ', CHUNK)
    assert verify_quote('“…수요자를 기만할 염려가 있는 상표를 등록받을 수 없다...”', CHUNK)
    assert verify_quote('"수요자를 기만할 염려가 있는 상표를"', CHUNK.replace(' ', '\n  '))


def test_verify_quote_rejects_short_empty_and_invented_quotes():
    assert not verify_quote('', CHUNK)
    assert not verify_quote('“상표를 등록받을”', CHUNK)            # 8 chars after normalization
    assert not verify_quote('신용을 이용하여 부당한 이익을 얻을 목적이 있어야 한다', CHUNK)   # from the answer
    assert not verify_quote('수요자를 기만할 염려가 있는 상표는 등록받을 수 없다', CHUNK)    # paraphrase


def test_quote_beyond_the_shown_truncation_is_not_verified(tmp_path):
    hidden = '이 문장은 잘린 뒤에만 나오는 법리 설명이다'
    text = '가' * 3500 + hidden
    model = ScriptedModel([{'grade': 2, 'reason': '', 'evidence_quote': hidden}])
    store = JudgmentStore(tmp_path / 'j.jsonl')
    judge_pool([pair(text=text)], model, store, think=False)
    row = store.rows(judge_key(model, think=False))[('c1', 'k1')]
    assert row['grade'] == 1 and row['details']['quote_verified'] is False


def test_unverified_grade_2_is_downgraded_and_details_are_stored(tmp_path):
    store = JudgmentStore(tmp_path / 'j.jsonl')
    invented = '신용을 이용하여 부당한 이익을 얻을 목적이 있어야 한다'
    verbatim = '수요자를 기만할 염려가 있는 상표를 등록받을 수 없다'
    model = ScriptedModel([{'grade': 2, 'reason': 'a', 'evidence_quote': invented},
                           {'grade': 2, 'reason': 'b', 'evidence_quote': verbatim}])
    judge_pool([pair('k1', text=CHUNK), pair('k2', text=CHUNK)], model, store, think=False)
    rows = JudgmentStore(tmp_path / 'j.jsonl').rows(judge_key(model, think=False))
    assert rows[('c1', 'k1')]['grade'] == 1
    assert rows[('c1', 'k1')]['details'] == {'model_grade': 2, 'evidence_quote': invented, 'quote_verified': False}
    assert rows[('c1', 'k2')]['grade'] == 2
    assert rows[('c1', 'k2')]['details'] == {'model_grade': 2, 'evidence_quote': verbatim, 'quote_verified': True}


def test_grades_0_and_1_are_not_changed_by_quote_verification(tmp_path):
    store = JudgmentStore(tmp_path / 'j.jsonl')
    model = ScriptedModel([{'grade': 0, 'reason': '', 'evidence_quote': ''},
                           {'grade': 1, 'reason': '', 'evidence_quote': '지어낸 문장이지만 충분히 긴 인용문입니다'}])
    judge_pool([pair('k1', text=CHUNK), pair('k2', text=CHUNK)], model, store, think=False)
    rows = store.rows(judge_key(model, think=False))
    assert rows[('c1', 'k1')]['grade'] == 0 and rows[('c1', 'k1')]['details']['model_grade'] == 0
    assert rows[('c1', 'k2')]['grade'] == 1 and rows[('c1', 'k2')]['details']['quote_verified'] is False


def test_parse_failure_stores_no_details(tmp_path):
    store = JudgmentStore(tmp_path / 'j.jsonl')
    bad = StructuredOutputError('bad', 'x')
    model = ScriptedModel([bad, bad])
    judge_pool([pair()], model, store, think=False)
    assert store.rows(judge_key(model, think=False))[('c1', 'k1')]['details'] is None


def test_schema_asks_for_the_quote_before_the_grade():
    assert list(RelevanceGrade.model_json_schema()['properties']) == ['evidence_quote', 'grade', 'reason']

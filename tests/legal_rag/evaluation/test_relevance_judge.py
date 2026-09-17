from legal_rag.evaluation.judgment_store import JudgmentStore
from legal_rag.evaluation.relevance_judge import (JUDGE_MAX_TOKENS, JUDGE_TEMPERATURE, RelevanceGrade, judge_key,
                                                  judge_pool)
from legal_rag.llm.base import ModelResponse, StructuredOutputError


class ScriptedModel:
    name = 'scripted'

    def __init__(self, replies):
        self.replies, self.messages, self.settings = list(replies), [], []

    def generate(self, messages, *, response_schema=None, temperature=0.2, max_tokens=1024):
        self.messages.append(messages)
        self.settings.append((response_schema, temperature, max_tokens))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ModelResponse(text='', parsed=RelevanceGrade(**reply))


def pair(chunk_id='k1', **extra):
    fields = dict(case_id='c1', chunk_id=chunk_id, document_id='d1', document_type='judgment',
                  title='상표법위반', text='부정경쟁의 목적이란 …', question='부정경쟁의 목적은?',
                  reference_answer='신용을 이용하여 부당한 이익을 얻을 목적', ranks={'hybrid': 3},
                  is_gold_chunk=True)
    fields.update(extra)
    return fields


def test_grades_are_appended_and_resume_skips_judged(tmp_path):
    store = JudgmentStore(tmp_path / 'j.jsonl')
    model = ScriptedModel([{'grade': 2, 'reason': '법리 명시'}])
    assert judge_pool([pair()], model, store) == {'judged': 1, 'skipped': 0, 'failed': 0}
    again = judge_pool([pair()], ScriptedModel([]), store)
    assert again == {'judged': 0, 'skipped': 1, 'failed': 0}
    assert store.grades(judge_key(model)) == {('c1', 'k1'): 2}


def test_prompt_hides_rank_mode_and_gold_status(tmp_path):
    model = ScriptedModel([{'grade': 1, 'reason': ''}])
    judge_pool([pair()], model, JudgmentStore(tmp_path / 'j.jsonl'))
    prompt = ' '.join(message.content for message in model.messages[0])
    assert '부정경쟁의 목적은?' in prompt and '부당한 이익' in prompt and '상표법위반' in prompt
    for leaked in ('hybrid', 'gold', 'rank', 'is_gold_chunk'):
        assert leaked not in prompt


def test_malformed_output_retries_once_then_records_null_not_zero(tmp_path):
    store = JudgmentStore(tmp_path / 'j.jsonl')
    bad = StructuredOutputError('bad', 'not json')
    model = ScriptedModel([bad, bad])
    assert judge_pool([pair()], model, store) == {'judged': 0, 'skipped': 0, 'failed': 1}
    assert len(model.messages) == 2
    assert store.grades(judge_key(model)) == {}
    assert store.failures(judge_key(model)) == [('c1', 'k1')]


def test_retry_success_counts_as_judged(tmp_path):
    model = ScriptedModel([StructuredOutputError('bad', 'x'), {'grade': 0, 'reason': ''}])
    assert judge_pool([pair()], model, JudgmentStore(tmp_path / 'j.jsonl'))['judged'] == 1


def test_long_chunks_are_truncated_in_the_prompt(tmp_path):
    model = ScriptedModel([{'grade': 0, 'reason': ''}])
    judge_pool([pair(text='가' * 10_000)], model, JudgmentStore(tmp_path / 'j.jsonl'))
    assert '가' * 3000 not in ' '.join(m.content for m in model.messages[0])


def test_braces_in_chunk_text_and_title_are_passed_through(tmp_path):
    model = ScriptedModel([{'grade': 0, 'reason': ''}])
    judge_pool([pair(text='조건 {a} 및 {0} }{', title='{title}')], model, JudgmentStore(tmp_path / 'j.jsonl'))
    user = model.messages[0][1].content
    assert '조건 {a} 및 {0} }{' in user and '{title}' in user


def test_transport_errors_propagate_without_recording(tmp_path):
    import pytest
    store = JudgmentStore(tmp_path / 'j.jsonl')
    with pytest.raises(TimeoutError):
        judge_pool([pair()], ScriptedModel([TimeoutError('slow')]), store)
    assert not store.has(judge_key(ScriptedModel([])), 'c1', 'k1')


def test_generate_uses_the_hashed_generation_settings(tmp_path):
    model = ScriptedModel([{'grade': 1, 'reason': ''}])
    judge_pool([pair()], model, JudgmentStore(tmp_path / 'j.jsonl'))
    assert model.settings == [(RelevanceGrade, JUDGE_TEMPERATURE, JUDGE_MAX_TOKENS)]

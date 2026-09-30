import pytest

from legal_rag.training.adapter_eval import (
    case_contexts, fit_retrieved, pairwise_answers, pending_generations, prompt_tokens, retrieved_messages,
    sample_cases)
from legal_rag.training.dataset import DOMAINS, messages_for


class Tokenizer:
    """One token per character of a flat template, so budgets are easy to reason about."""

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {'tokenize': False, 'add_generation_prompt': True, 'enable_thinking': False}
        return '|'.join(m['content'] for m in messages) + '|A:'

    def __call__(self, text, **kwargs):
        assert kwargs['add_special_tokens'] is False
        return {'input_ids': list(range(len(text)))}


TOK = Tokenizer()


def hit(rank, text, *, document='d1', title='제목1', order=0):
    return {'rank': rank, 'chunk_id': f'{document}_c{order}', 'document_id': document, 'title': title,
            'order': order, 'text': text}


def test_one_document_is_the_training_format():
    assert retrieved_messages('질문?', [('제목1', ['가나다', '라마바'])]) == \
        messages_for('질문?', '가나다\n\n[발췌 구분]\n\n라마바', '제목1')


def test_several_documents_get_one_block_each_in_order():
    user = retrieved_messages('질문?', [('A', ['x']), ('B', ['y'])])[1]['content']
    assert user == '[법률 자료]\n제목: A\nx\n[/법률 자료]\n\n[법률 자료]\n제목: B\ny\n[/법률 자료]\n\n질문: 질문?'


def test_no_passages_is_the_empty_context_format():
    assert retrieved_messages('질문?', []) == messages_for('질문?', '')


def test_chunks_that_fit_are_used_whole_in_rank_order():
    hits = [hit(1, '상표 유사 판단', document='d2', title='B'), hit(2, '상표 등록 요건', document='d1', title='A')]
    fitted = fit_retrieved('상표', hits, tokenizer=TOK, budget=10_000)
    assert [p['chunk_id'] for p in fitted['passages']] == ['d2_c0', 'd1_c0']
    assert all(p['whole'] for p in fitted['passages'])
    assert fitted['messages'] == retrieved_messages('상표', [('B', ['상표 유사 판단']), ('A', ['상표 등록 요건'])])
    assert fitted['prompt_tokens'] == prompt_tokens(fitted['messages'], TOK)


def test_a_long_chunk_contributes_only_question_windows_within_budget():
    filler = '무관한 내용입니다. ' * 40
    text = filler + '상표의 유사 여부는 외관 호칭 관념으로 판단한다. ' + filler
    budget = prompt_tokens(retrieved_messages('상표 유사', []), TOK) + 200
    fitted = fit_retrieved('상표 유사', [hit(1, text)], tokenizer=TOK, budget=budget, window_chars=64)
    assert fitted['prompt_tokens'] <= budget
    assert fitted['passages'] and not any(p['whole'] for p in fitted['passages'])
    used = ''.join(text[p['start']:p['end']] for p in fitted['passages'])
    assert '상표' in used


def test_nothing_fits_falls_back_to_the_empty_context():
    budget = prompt_tokens(retrieved_messages('상표', []), TOK)
    fitted = fit_retrieved('상표', [hit(1, '상표 ' * 50)], tokenizer=TOK, budget=budget, window_chars=32)
    assert fitted['passages'] == [] and fitted['messages'] == messages_for('상표', '')


def test_windows_without_question_terms_are_never_used():
    fitted = fit_retrieved('특허', [hit(1, '무관한 문장. ' * 30)], tokenizer=TOK,
                           budget=prompt_tokens(retrieved_messages('특허', []), TOK) + 60, window_chars=32)
    assert fitted['passages'] == []


def test_passages_of_one_document_follow_source_order():
    hits = [hit(1, '상표 둘', order=1), hit(2, '상표 하나', order=0)]
    fitted = fit_retrieved('상표', hits, tokenizer=TOK, budget=10_000)
    assert fitted['messages'][1]['content'].index('상표 하나') < fitted['messages'][1]['content'].index('상표 둘')


def test_budget_below_the_empty_prompt_is_refused():
    with pytest.raises(ValueError, match='budget'):
        fit_retrieved('상표', [], tokenizer=TOK, budget=5)


def row_for(case='c1', domain='civil', document='d1'):
    oracle = messages_for('질문?', '원문', '제목')
    return {'id': case, 'domain': domain, 'document_id': document, 'question': '질문?', 'answer': '참조',
            'messages': oracle + [{'role': 'assistant', 'content': '참조'}]}


def test_case_contexts_hold_three_conditions_and_the_gold_flag():
    row = row_for()
    context = case_contexts(row, [hit(1, '질문 관련', document='d9'), hit(2, '질문 근거', document='d1')],
                            tokenizer=TOK, budget=10_000)
    assert context['case_id'] == 'c1' and context['reference_answer'] == '참조'
    assert context['gold_retrieved'] is True and context['gold_rank'] == 2
    assert context['conditions']['oracle']['messages'] == row['messages'][:-1]
    assert context['conditions']['none']['messages'] == messages_for('질문?', '')
    assert [r['chunk_id'] for r in context['retrieved']] == ['d9_c0', 'd1_c0']
    assert 'text' not in context['retrieved'][0]


def test_sampling_is_balanced_and_deterministic():
    rows = [row_for(f'{d}-{i}', d) for d in DOMAINS for i in range(10)]
    first = sample_cases(rows, per_domain=3, seed=7)
    assert first == sample_cases(list(reversed(rows)), per_domain=3, seed=7)
    assert sorted(r['domain'] for r in first) == sorted(DOMAINS * 3)
    with pytest.raises(ValueError, match='civil'):
        sample_cases([r for r in rows if r['domain'] != 'civil'] + rows[:1], per_domain=3, seed=7)


def test_pending_generations_skip_done_keys():
    contexts = [{'case_id': 'c1', 'conditions': {'none': {}, 'retrieved': {}}},
                {'case_id': 'c2', 'conditions': {'none': {}, 'retrieved': {}}}]
    done = {('c1', 'none', 'base'), ('c1', 'none', 'adapter')}
    todo = pending_generations(contexts, done, conditions=['none'], models=['base', 'adapter'])
    assert [(c['case_id'], condition, model) for c, condition, model in todo] == \
        [('c2', 'none', 'base'), ('c2', 'none', 'adapter')]


def answer(case, condition, model, text='답', **extra):
    return {'case_id': case, 'condition': condition, 'model': model, 'text': text, **extra}


def test_pairwise_answers_pair_base_and_adapter_for_one_condition():
    contexts = [{'case_id': 'c1', 'question': 'q', 'reference_answer': 'ref'}]
    answers = [answer('c1', 'retrieved', 'base', 'b'), answer('c1', 'retrieved', 'adapter', 'a'),
               answer('c1', 'none', 'base', 'x')]
    assert pairwise_answers(contexts, answers, condition='retrieved') == \
        [{'case_id': 'c1', 'question': 'q', 'reference_answer': 'ref', 'base': 'b', 'adapter': 'a'}]


def test_pairwise_answers_refuse_a_missing_or_failed_side():
    contexts = [{'case_id': 'c1', 'question': 'q', 'reference_answer': 'ref'}]
    with pytest.raises(ValueError, match='c1'):
        pairwise_answers(contexts, [answer('c1', 'retrieved', 'base')], condition='retrieved')
    with pytest.raises(ValueError, match='c1'):
        pairwise_answers(contexts, [answer('c1', 'retrieved', 'base'),
                                    answer('c1', 'retrieved', 'adapter', error='CUDA OOM')], condition='retrieved')

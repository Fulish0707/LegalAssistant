import pytest

from legal_rag.training.tokenization import encode_example, pad_batch, balance_encoded


class Tokenizer:
    eos_token_id = 999
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is False
        assert kwargs['add_generation_prompt'] is True
        return 'PROMPT:' + messages[1]['content'] + ':ASSISTANT:<think>\n\n</think>\n\n'

    def __call__(self, text, **kwargs):
        assert kwargs['add_special_tokens'] is False
        return {'input_ids': [ord(c) for c in text]}


def row(answer='answer'):
    return {'messages': [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'source question'},
                         {'role': 'assistant', 'content': answer}], 'answer': answer,
            'domain': 'civil', 'id': 'one'}


def test_masks_prompt_and_supervises_complete_answer_and_eos():
    encoded = encode_example(row(), Tokenizer(), max_length=200)
    supervised = [x for x in encoded['labels'] if x != -100]
    assert supervised == [ord(x) for x in 'answer'] + [999]
    assert encoded['input_ids'][-1] == 999
    assert encoded['labels'][0] == -100
    assert len(encoded['labels']) == len(encoded['input_ids'])


def test_overlength_drops_whole_example_instead_of_truncating_answer():
    assert encode_example(row('x' * 200), Tokenizer(), max_length=100) is None
    with pytest.raises(ValueError, match='empty'):
        encode_example(row('  '), Tokenizer(), max_length=100)


def test_padding_masks_only_padding_not_real_eos():
    batch = pad_batch([{'input_ids': [4, 999], 'labels': [-100, 999]},
                       {'input_ids': [5, 6, 999], 'labels': [-100, 6, 999]}], pad_token_id=999)
    assert batch['input_ids'][0] == [4, 999, 999]
    assert batch['labels'][0] == [-100, 999, -100]
    assert batch['attention_mask'][0] == [1, 1, 0]


def test_token_filter_rebalances_and_rejects_empty_domain():
    examples = [dict(row(), domain='civil', id=str(i)) for i in range(3)]
    examples.append(dict(row(), domain='criminal', id='c'))
    encoded, report = balance_encoded(examples, Tokenizer(), max_length=200, domains=('civil','criminal'))
    assert len(encoded) == 2
    assert report['selected_per_domain'] == 1
    examples[-1] = dict(examples[-1], messages=row('x' * 300)['messages'])
    with pytest.raises(ValueError, match='criminal'):
        balance_encoded(examples, Tokenizer(), max_length=200, domains=('civil','criminal'))

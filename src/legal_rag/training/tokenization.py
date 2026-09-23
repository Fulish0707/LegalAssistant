"""Answer-only causal-LM labels. Tokenizer is injected; no torch import on preparation."""
from collections import Counter

from .dataset import DOMAINS, digest


def prompt_text(example, tokenizer):
    messages = example['messages']
    if [m['role'] for m in messages] != ['system', 'user', 'assistant']:
        raise ValueError('expected one system, user and assistant turn')
    return tokenizer.apply_chat_template(messages[:-1], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def encode_example(example, tokenizer, *, max_length=1536):
    answer = example['messages'][-1]['content'].strip()
    if not answer:
        raise ValueError('empty answer')
    if tokenizer.eos_token_id is None:
        raise ValueError('tokenizer must define EOS')
    # Encode separately so training exactly matches the inference generation
    # prefix. The empty non-thinking block belongs to the masked prompt.
    prompt = tokenizer(prompt_text(example, tokenizer), add_special_tokens=False)['input_ids']
    completion = tokenizer(answer, add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
    if len(prompt) + len(completion) > max_length:
        return None
    return {'input_ids': prompt + completion, 'labels': [-100] * len(prompt) + completion}


def pad_batch(features, *, pad_token_id):
    if not features:
        raise ValueError('empty batch')
    width = max(len(f['input_ids']) for f in features)
    batch = {'input_ids': [], 'labels': [], 'attention_mask': []}
    for feature in features:
        n = len(feature['input_ids'])
        batch['input_ids'].append(feature['input_ids'] + [pad_token_id] * (width - n))
        batch['labels'].append(feature['labels'] + [-100] * (width - n))
        batch['attention_mask'].append([1] * n + [0] * (width - n))
    return batch


def balance_encoded(examples, tokenizer, *, max_length=1536, domains=DOMAINS, seed=42):
    buckets = {d: [] for d in domains}
    dropped = Counter()
    for example in examples:
        domain = example['domain']
        if domain not in buckets:
            raise ValueError(f'unexpected domain: {domain}')
        encoded = encode_example(example, tokenizer, max_length=max_length)
        if encoded is None:
            dropped[domain] += 1
        else:
            buckets[domain].append((example['id'], encoded))
    empty = [d for d, rows in buckets.items() if not rows]
    if empty:
        raise ValueError(f'no examples fit the token limit for {empty}')
    count = min(len(rows) for rows in buckets.values())
    selected = []
    for domain in domains:
        selected.extend(sorted(buckets[domain], key=lambda x: digest(f'{seed}:{x[0]}'))[:count])
    selected.sort(key=lambda x: digest(f'{seed}:{x[0]}'))
    return [encoded for _, encoded in selected], {
        'max_length': max_length, 'overlength_by_domain': dict(dropped),
        'eligible_by_domain': {d: len(rows) for d, rows in buckets.items()},
        'selected_per_domain': count, 'selected_ids': [identifier for identifier, _ in selected],
    }

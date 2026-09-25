"""Evaluate the legal SFT adapter on retrieved evidence, in the format it was trained on.

The adapter learned ``training.dataset.messages_for``: one ``[법률 자료]`` block of verbatim
passages, then the question, answered in plain text. Every condition here keeps that format and
changes only where the passages come from, so a difference between base and adapter is the
adapter's, not a format change's.

Retrieved chunks are far longer than any training prompt, so they are fitted into a token budget
with the same question-only passage selector that built the training set. Nothing here imports
torch or transformers; the tokenizer is injected.
"""
from __future__ import annotations

import random
from typing import Iterable

from legal_rag.training.dataset import DOMAINS, SYSTEM, messages_for
from legal_rag.training.full_sft import select_passages
from legal_rag.training.tokenization import generation_prompt

CONDITIONS = ('none', 'retrieved', 'oracle')
MODELS = ('base', 'adapter')
SEPARATOR = '\n\n[발췌 구분]\n\n'


def prompt_tokens(messages: list[dict], tokenizer) -> int:
    return len(tokenizer(generation_prompt(messages, tokenizer), add_special_tokens=False)['input_ids'])


def retrieved_messages(question: str, blocks: list[tuple[str, list[str]]]) -> list[dict]:
    """``blocks`` is one (title, passages) pair per document, in the order they are shown.

    One document reproduces ``messages_for`` exactly; none reproduces its empty-context prompt."""
    if not blocks:
        return messages_for(question, '')
    user = '\n\n'.join(f'[법률 자료]\n제목: {title or "제목 없음"}\n{SEPARATOR.join(passages)}\n[/법률 자료]'
                       for title, passages in blocks)
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': f'{user}\n\n질문: {question}'}]


def _blocks(pieces: list[dict], titles: dict[str, str]) -> list[tuple[str, list[str]]]:
    # Documents in the rank order of their first piece; a document's pieces in source order.
    documents = list(dict.fromkeys(p['document_id'] for p in sorted(pieces, key=lambda p: p['rank'])))
    return [(titles[doc], [p['text'] for p in sorted((p for p in pieces if p['document_id'] == doc),
                                                      key=lambda p: (p['order'], p['start']))])
            for doc in documents]


def fit_retrieved(question: str, hits: list[dict], *, tokenizer, budget: int, window_chars: int = 480) -> dict:
    """Fill the prompt with retrieved text, best-ranked chunk first, without exceeding ``budget``.

    ``hits`` carry ``rank``, ``chunk_id``, ``document_id``, ``title``, ``order`` and ``text``. A chunk
    that fits is taken whole; otherwise its ``select_passages`` windows that share a term with the
    question are added best first while they fit."""
    if prompt_tokens(messages_for(question, ''), tokenizer) > budget:
        raise ValueError(f'budget {budget} is smaller than the prompt without any passage')
    titles = {hit['document_id']: hit['title'] for hit in hits}
    pieces: list[dict] = []

    def fits(candidate: dict) -> bool:
        return prompt_tokens(retrieved_messages(question, _blocks(pieces + [candidate], titles)), tokenizer) <= budget

    for hit in sorted(hits, key=lambda h: h['rank']):
        base = {key: hit[key] for key in ('rank', 'chunk_id', 'document_id', 'order')}
        whole = dict(base, start=0, end=len(hit['text']), text=hit['text'], whole=True)
        if fits(whole):
            pieces.append(whole)
            continue
        taken: list[dict] = []
        for passage in select_passages(question, hit['text'], window_chars=window_chars):
            if passage['score'] <= 0:
                break
            if any(passage['start'] < p['end'] and p['start'] < passage['end'] for p in taken):
                continue
            candidate = dict(base, start=passage['start'], end=passage['end'], text=passage['text'], whole=False)
            if fits(candidate):
                pieces.append(candidate)
                taken.append(candidate)
    messages = retrieved_messages(question, _blocks(pieces, titles))
    return {'messages': messages, 'prompt_tokens': prompt_tokens(messages, tokenizer),
            'passages': [{key: p[key] for key in ('rank', 'chunk_id', 'document_id', 'start', 'end', 'whole')}
                         for p in pieces]}


def case_contexts(row: dict, hits: list[dict], *, tokenizer, budget: int, window_chars: int = 480) -> dict:
    """Everything the generator needs for one test row: finished messages for every condition."""
    if [m['role'] for m in row['messages']] != ['system', 'user', 'assistant']:
        raise ValueError(f"{row['id']}: expected one system, user and assistant turn")
    oracle = row['messages'][:-1]
    none = messages_for(row['question'], '')
    gold = next((h['rank'] for h in sorted(hits, key=lambda h: h['rank']) if h['document_id'] == row['document_id']),
                None)
    return {
        'case_id': row['id'], 'domain': row['domain'], 'document_id': row['document_id'],
        'question': row['question'], 'reference_answer': row['answer'],
        'retrieved': [{key: h[key] for key in ('rank', 'chunk_id', 'document_id', 'title')} for h in hits],
        'gold_retrieved': gold is not None, 'gold_rank': gold,
        'conditions': {
            'none': {'messages': none, 'prompt_tokens': prompt_tokens(none, tokenizer)},
            'retrieved': fit_retrieved(row['question'], hits, tokenizer=tokenizer, budget=budget,
                                       window_chars=window_chars),
            'oracle': {'messages': oracle, 'prompt_tokens': prompt_tokens(oracle, tokenizer)},
        },
    }


def sample_cases(rows: Iterable[dict], *, per_domain: int, seed: int) -> list[dict]:
    """``per_domain`` rows from each domain, independent of input order."""
    by_domain: dict[str, list[dict]] = {domain: [] for domain in DOMAINS}
    for row in rows:
        by_domain.setdefault(row['domain'], []).append(row)
    chosen = []
    for domain in DOMAINS:
        candidates = sorted(by_domain[domain], key=lambda r: r['id'])
        if len(candidates) < per_domain:
            raise ValueError(f'{domain}: {len(candidates)} rows, fewer than {per_domain}')
        chosen += random.Random(f'{seed}:{domain}').sample(candidates, per_domain)
    return sorted(chosen, key=lambda r: (DOMAINS.index(r['domain']), r['id']))


def pending_generations(contexts: list[dict], done: set[tuple[str, str, str]], *, conditions, models) -> list:
    """(context, condition, model) triples not yet in ``done``, in a stable order."""
    return [(context, condition, model) for context in contexts for condition in conditions
            for model in models if (context['case_id'], condition, model) not in done]


def pairwise_answers(contexts: list[dict], answers: list[dict], *, condition: str) -> list[dict]:
    """Rows for ``answer_compare.pairwise_rows``: one per case, base and adapter as the two sides."""
    texts = {(a['case_id'], a['model']): a for a in answers if a['condition'] == condition}
    rows = []
    for context in contexts:
        pair = {}
        for model in MODELS:
            found = texts.get((context['case_id'], model))
            if found is None or found.get('error'):
                reason = 'missing' if found is None else found['error']
                raise ValueError(f"{context['case_id']}: no {model} answer under {condition} ({reason})")
            pair[model] = found['text']
        rows.append({'case_id': context['case_id'], 'question': context['question'],
                     'reference_answer': context['reference_answer'], **pair})
    return rows

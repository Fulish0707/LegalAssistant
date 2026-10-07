"""Grade pooled (question, chunk) pairs with a chat model.

The prompt shows the question, the reference answer and the chunk only — never the
retrieval mode, rank, or whether the chunk belongs to the gold document.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Literal

import httpx
from pydantic import BaseModel

from legal_rag.llm.base import ChatModel, Message, StructuredOutputError

from .judgment_store import JudgeKey, JudgmentStore
from .relevance_prompts import (ACTIVE_CORPUS_DOMAINS, DEFAULT_PROMPT_PROFILE, GUIDANCE, ROUTING_VERSION,
                                guidance_domain, question_domain, system_prompt, validate_profile)

MAX_CHUNK_CHARS = 3500
MAX_ANSWER_CHARS = 800
JUDGE_TEMPERATURE = 0.0
JUDGE_ATTEMPTS = 2
JUDGE_THINK = False   # CLI default; think is a run parameter and part of the key.
# Ollama counts thinking tokens in num_predict, so think on needs far more room. The quote needs
# room even without thinking. num_ctx stays 8192 for both: qwen3:8b already reported 7.3GB there,
# and a larger context risks spilling off an 8GB GPU.
JUDGE_GENERATION = {False: {'max_tokens': 512, 'num_ctx': 8192}, True: {'max_tokens': 4096, 'num_ctx': 8192}}
# Bump QUOTE_RULE_VERSION whenever verify_quote's normalization or the downgrade rule changes.
QUOTE_RULE_VERSION = 2
MIN_QUOTE_CHARS = 15
QUOTE_SPAN_RULE = r'(?<=다\.)[ \t]+|\n+'
QUOTE_RETRY_ATTEMPTS = 1

SYSTEM = """당신은 한국 법률 검색 평가자입니다. 질문, 참조 답변, 검색된 문서 조각이 주어집니다.
문서 조각이 질문의 법적 쟁점에 답하는 근거인지 등급을 매기십시오.
참조 답변은 기대되는 법리를 알려 주는 힌트일 뿐이며 오류(예: 틀린 조문 번호)가 있을 수 있습니다. 조각이 참조 답변의 잘못된 세부 사항까지 뒷받침할 필요는 없습니다.

2 = 조각 자체가 질문의 쟁점에 답하는 법리·기준·요건을 직접 명시함. 다른 사건이라도 같은 법리를 명시하면 2입니다.
1 = 같은 법적 쟁점이나 조문을 다루지만 법리가 그것만으로 답할 만큼 충분히 명시되지 않음. 또는 관련 분야에서 법리를 명시하지 않고 사실관계에 적용만 함.
0 = 다른 법적 쟁점·제도·절차·권리를 다룸(예: 보정과 정정, 저작권과 특허권, 출원 거절과 등록 후 무효·정정), 또는 용어만 겹침.

당사자의 주장만 있고 판단기관이 이를 채택한 독립적인 판단이 없으면 2로 매기지 마십시오. 같은 쟁점의 주장이나 사례는 1입니다.
주장과 판단이 섞인 조각에서는 독립적인 판단 문장이 법리·요건을 직접 밝히면 2가 가능합니다.
같은 쟁점의 사실관계·적용 사례에 일반적인 기준이 없으면 1입니다. 단순히 참조 답변의 조문 번호가 없거나 다르다는 이유로 0을 주지 마십시오.
출원 중 보정과 등록 후 정정, 국내 절차와 국제출원의 특정 절차처럼 적용 범위를 구별하십시오. 질문과 다른 절차의 요건은 답의 직접 근거가 아닙니다.
범죄의 적용 법조를 묻는 질문에서는 해당 범죄에 적용되는 조문을 명시한 판단도 직접 근거입니다. 반면 형량·사건 사실만으로 구성요건을 추정하지 마십시오.

등급을 정하기 전에 [문서 조각]에서 등급을 가장 잘 뒷받침하는 문장 하나를 글자 그대로 복사하여 evidence_quote에 넣으십시오.
질문이나 참조 답변에서 복사하지 마십시오. 관련 내용이 없으면 빈 문자열로 두십시오.

JSON으로만 답하십시오: {"evidence_quote": "조각에서 그대로 복사한 문장", "grade": 0|1|2, "reason": "한 문장 근거"}"""

USER = """[질문]
{question}

[참조 답변]
{answer}

[문서 조각] ({document_type}) {title}
{text}"""


class RelevanceGrade(BaseModel):
    evidence_quote: str
    grade: Literal[0, 1, 2]
    reason: str


class QuoteSelection(BaseModel):
    span_index: int


QUOTE_SYSTEM = """법률 평가에서 이미 매긴 등급을 변경하지 말고 근거 문장만 선택하십시오.
질문에 대한 법리·기준·요건 또는 적용 법조를 직접 뒷받침하는 원문 후보의 번호를 고르십시오.
당사자의 주장만을 판단기관의 판단으로 취급하지 마십시오. 적절한 후보가 없으면 0을 선택하십시오.
JSON으로만 답하십시오: {"span_index": 정수}"""
QUOTE_USER = """[질문]
{question}
[기존 판단 이유]
{reason}
[복사 검증에 실패한 인용]
{quote}
[원문 후보]
{spans}"""


def generation_settings(*, think: bool) -> dict[str, int]:
    """max_tokens and num_ctx for one think mode; both are part of the prompt hash."""
    return dict(JUDGE_GENERATION[bool(think)])


def prompt_sha256(*, think: bool, prompt_profile: str = DEFAULT_PROMPT_PROFILE, domain: str | None = None) -> str:
    """Hash of everything that shapes a grade: prompt, schema, truncation, generation settings,
    thinking mode and the quote verification rule."""
    validate_profile(prompt_profile, domain)
    settings = generation_settings(think=think)
    shared_hash = hashlib.sha256(
        json.dumps([SYSTEM, USER, RelevanceGrade.model_json_schema(), MAX_CHUNK_CHARS, MAX_ANSWER_CHARS,
                    JUDGE_TEMPERATURE, settings['max_tokens'], JUDGE_ATTEMPTS, settings['num_ctx'], bool(think),
                    QUOTE_RULE_VERSION, MIN_QUOTE_CHARS, QUOTE_SYSTEM, QUOTE_USER,
                    QuoteSelection.model_json_schema(), QUOTE_SPAN_RULE, QUOTE_RETRY_ATTEMPTS],
                   ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if prompt_profile == 'shared':
        return shared_hash
    override = question_domain({}, domain) if domain not in {'mixed', 'unknown'} else domain
    return hashlib.sha256(json.dumps([shared_hash, prompt_profile, GUIDANCE, ROUTING_VERSION, override,
                                     sorted(ACTIVE_CORPUS_DOMAINS)],
                                    ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def judge_key(model: ChatModel | str, *, think: bool, prompt_profile: str = DEFAULT_PROMPT_PROFILE,
              domain: str | None = None) -> JudgeKey:
    """The judgment key for a model (or its name) in one think mode; every script builds keys here."""
    return JudgeKey(model=model if isinstance(model, str) else model.name,
                    prompt_sha256=prompt_sha256(think=think, prompt_profile=prompt_profile, domain=domain))


def describe_key(key: JudgeKey, *, think: bool) -> str:
    return f'judge {key.model} prompt {key.prompt_sha256[:12]} think {"on" if think else "off"}'


_WHITESPACE = re.compile(r'\s+')
_QUOTE_EDGES = '\'"“”‘’…'


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(' ', text).strip()


def _strip_quote_edges(text: str) -> str:
    previous = None
    while text != previous:
        previous = text
        text = text.strip().strip(_QUOTE_EDGES).removeprefix('...').removesuffix('...')
    return text.strip()


def verify_quote(quote: str, chunk_text: str) -> bool:
    """True when the quote, after collapsing whitespace and stripping surrounding quote marks and
    ellipses, is at least MIN_QUOTE_CHARS long and occurs in the chunk text as passed.

    Callers pass the chunk truncated as the judge saw it."""
    normalized = _strip_quote_edges(_normalize(quote or ''))
    return len(normalized) >= MIN_QUOTE_CHARS and normalized in _normalize(chunk_text)


def _messages(pair: dict, *, prompt_profile: str = DEFAULT_PROMPT_PROFILE, domain: str | None = None) -> list[Message]:
    user = USER.format(question=pair['question'], answer=(pair['reference_answer'] or '')[:MAX_ANSWER_CHARS],
                       document_type=pair['document_type'], title=pair.get('title') or '',
                       text=pair['text'][:MAX_CHUNK_CHARS])
    return [Message(role='system', content=system_prompt(SYSTEM, pair, profile=prompt_profile, domain=domain)),
            Message(role='user', content=user)]


def _repair_quote(pair: dict, parsed: RelevanceGrade, model: ChatModel, *, max_tokens: int) -> dict:
    """One quote-only selection call; copy the chosen span directly from the shown source."""
    text = pair['text'][:MAX_CHUNK_CHARS]
    spans = [span.strip() for span in re.split(QUOTE_SPAN_RULE, text)
             if verify_quote(span.strip(), text)]
    audit = {'quote_retry_attempted': True, 'quote_retry_succeeded': False,
             'quote_retry_span_index': None, 'quote_retry_error': None}
    if not spans:
        audit['quote_retry_error'] = 'no_candidate_spans'
        return audit
    messages = [Message(role='system', content=QUOTE_SYSTEM), Message(role='user', content=QUOTE_USER.format(
        question=pair['question'], reason=parsed.reason, quote=parsed.evidence_quote,
        spans='\n'.join(f'{index}: {span}' for index, span in enumerate(spans, 1))))]
    try:
        selection = model.generate(messages, response_schema=QuoteSelection,
                                   temperature=JUDGE_TEMPERATURE, max_tokens=max_tokens).parsed
    except StructuredOutputError as exc:
        audit.update(quote_retry_error='structured_output_error', quote_retry_raw=exc.text)
        return audit
    index = selection.span_index
    audit['quote_retry_span_index'] = index
    if not 1 <= index <= len(spans):
        audit['quote_retry_error'] = 'no_support' if index == 0 else 'invalid_span_index'
        return audit
    quote = spans[index - 1]
    audit.update(quote_retry_succeeded=verify_quote(quote, text), evidence_quote=quote)
    return audit


def judge_pool(pairs: Iterable[dict], model: ChatModel, store: JudgmentStore, *, think: bool,
               progress: Callable[[str], None] | None = None, retry_failed: bool = False,
               prompt_profile: str = DEFAULT_PROMPT_PROFILE, domain: str | None = None) -> dict[str, int]:
    """Judge each unjudged pair, making up to JUDGE_ATTEMPTS tries on schema failures.

    An invalid grade-2 quote gets one source-span selection retry before downgrade to 1.
    Original and final quotes and the model's grade are preserved in ``details``.
    When every attempt fails, the pair is stored with grade None, details None and the raw text.
    Transport errors are not caught: the run stops and a rerun skips pairs already in the store.
    With ``retry_failed``, pairs whose latest row is a failure are judged again and a new row is appended.
    """
    model_think = getattr(model, 'think', None)
    if model_think is not None and bool(model_think) != bool(think):
        raise ValueError(f'model think={model_think} but judge_pool think={think}; the key would be wrong')
    key = judge_key(model, think=think, prompt_profile=prompt_profile, domain=domain)
    max_tokens = generation_settings(think=think)['max_tokens']
    counts = {'judged': 0, 'skipped': 0, 'failed': 0}
    pairs = list(pairs)
    domains_by_pair = {}
    for pair in pairs:
        identity = (pair['case_id'], pair['chunk_id'])
        selected = question_domain(pair, domain) if prompt_profile != 'shared' else None
        if identity in domains_by_pair and domains_by_pair[identity] != selected:
            raise ValueError(f'conflicting question domains for {identity}')
        domains_by_pair[identity] = selected
    existing = store.rows(key)
    for number, pair in enumerate(pairs, 1):
        selected_domain = question_domain(pair, domain) if prompt_profile != 'shared' else None
        cached = existing.get((pair['case_id'], pair['chunk_id']))
        if cached and cached['grade'] is not None and prompt_profile != 'shared' and (
                (cached['details'] or {}).get('question_domain') != selected_domain):
            raise ValueError(f"cached question-domain routing changed for {pair['case_id']}|{pair['chunk_id']}")
        if store.has(key, pair['case_id'], pair['chunk_id']) and not (
                retry_failed and store.is_failed(key, pair['case_id'], pair['chunk_id'])):
            counts['skipped'] += 1
            continue
        grade, reason, raw, details = None, '', None, None
        for _attempt in range(JUDGE_ATTEMPTS):
            try:
                parsed = model.generate(_messages(pair, prompt_profile=prompt_profile, domain=domain), response_schema=RelevanceGrade,
                                        temperature=JUDGE_TEMPERATURE, max_tokens=max_tokens).parsed
            except StructuredOutputError as exc:
                raw = exc.text
                continue
            verified = verify_quote(parsed.evidence_quote, pair['text'][:MAX_CHUNK_CHARS])
            reason = parsed.reason
            details = {'model_grade': parsed.grade, 'evidence_quote': parsed.evidence_quote,
                       'quote_verified': verified, 'original_evidence_quote': parsed.evidence_quote,
                       'original_quote_verified': verified, 'quote_retry_attempted': False,
                       'quote_retry_succeeded': False, 'prompt_profile': prompt_profile,
                       'question_domain': selected_domain,
                       'guidance_domain': guidance_domain(pair, profile=prompt_profile, domain=domain)}
            if parsed.grade == 2 and not verified:
                details.update(_repair_quote(pair, parsed, model, max_tokens=max_tokens))
                verified = bool(details['quote_retry_succeeded'])
                details['quote_verified'] = verified
            grade = 1 if parsed.grade == 2 and not verified else parsed.grade
            break
        store.append(key, case_id=pair['case_id'], chunk_id=pair['chunk_id'], document_id=pair['document_id'],
                     grade=grade, reason=reason, raw=None if grade is not None else raw, details=details)
        counts['judged' if grade is not None else 'failed'] += 1
        if progress and number % 25 == 0:
            progress(f'{number}/{len(pairs)} {counts}')
    return counts


def ollama_model_digest(base_url: str, model: str, *, transport: httpx.BaseTransport | None = None) -> str | None:
    """Best-effort digest of an Ollama model from ``GET /api/tags``; None if unavailable.

    Provenance only: the digest is recorded in judge_runs.jsonl, never in the judgment key."""
    try:
        with httpx.Client(base_url=base_url, timeout=5.0, transport=transport) as client:
            response = client.get('/api/tags')
            response.raise_for_status()
            models = response.json().get('models') or []
    except (httpx.HTTPError, ValueError, AttributeError):
        return None
    names = {model, model if ':' in model else f'{model}:latest'}
    return next((entry.get('digest') for entry in models if entry.get('name') in names), None)


def judge_run_record(key: JudgeKey, *, calibration_csv_sha256: str | None, calibration_only: bool,
                     retry_failed: bool, pairs: int, counts: dict[str, int], model_digest: str | None,
                     think: bool, prompt_profile: str = DEFAULT_PROMPT_PROFILE, domain: str | None = None) -> dict:
    return {'finished_at': datetime.now(timezone.utc).isoformat(), 'judge_model': key.model,
            'prompt_sha256': key.prompt_sha256, 'think': think, 'generation': generation_settings(think=think),
            'quote_rule_version': QUOTE_RULE_VERSION, 'model_digest': model_digest,
            'quote_retry_attempts': QUOTE_RETRY_ATTEMPTS, 'quote_span_rule': QUOTE_SPAN_RULE,
            'prompt_profile': prompt_profile, 'question_domain_override': domain, 'routing_version': ROUTING_VERSION,
            'active_corpus_domains': sorted(ACTIVE_CORPUS_DOMAINS),
            'calibration_csv_sha256': calibration_csv_sha256, 'calibration_only': calibration_only,
            'retry_failed': retry_failed, 'pairs': pairs, 'counts': counts}


def append_judge_run(pool_dir: Path, record: dict) -> Path:
    path = Path(pool_dir) / 'judge_runs.jsonl'
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    return path

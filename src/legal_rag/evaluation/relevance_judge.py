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
QUOTE_RULE_VERSION = 1
MIN_QUOTE_CHARS = 15

SYSTEM = """당신은 한국 법률 검색 평가자입니다. 질문, 참조 답변, 검색된 문서 조각이 주어집니다.
문서 조각이 질문의 법적 쟁점에 답하는 근거인지 등급을 매기십시오.
참조 답변은 기대되는 법리를 알려 주는 힌트일 뿐이며 오류(예: 틀린 조문 번호)가 있을 수 있습니다. 조각이 참조 답변의 잘못된 세부 사항까지 뒷받침할 필요는 없습니다.

2 = 조각 자체가 질문의 쟁점에 답하는 법리·기준·요건을 직접 명시함. 다른 사건이라도 같은 법리를 명시하면 2입니다.
1 = 같은 법적 쟁점이나 조문을 다루지만 법리가 그것만으로 답할 만큼 충분히 명시되지 않음. 또는 관련 분야에서 법리를 명시하지 않고 사실관계에 적용만 함.
0 = 다른 법적 쟁점·제도·절차·권리를 다룸(예: 보정과 정정, 저작권과 특허권, 출원 거절과 등록 후 무효·정정), 또는 용어만 겹침.

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


def generation_settings(*, think: bool) -> dict[str, int]:
    """max_tokens and num_ctx for one think mode; both are part of the prompt hash."""
    return dict(JUDGE_GENERATION[bool(think)])


def prompt_sha256(*, think: bool) -> str:
    """Hash of everything that shapes a grade: prompt, schema, truncation, generation settings,
    thinking mode and the quote verification rule."""
    settings = generation_settings(think=think)
    return hashlib.sha256(
        json.dumps([SYSTEM, USER, RelevanceGrade.model_json_schema(), MAX_CHUNK_CHARS, MAX_ANSWER_CHARS,
                    JUDGE_TEMPERATURE, settings['max_tokens'], JUDGE_ATTEMPTS, settings['num_ctx'], bool(think),
                    QUOTE_RULE_VERSION, MIN_QUOTE_CHARS],
                   ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def judge_key(model: ChatModel | str, *, think: bool) -> JudgeKey:
    """The judgment key for a model (or its name) in one think mode; every script builds keys here."""
    return JudgeKey(model=model if isinstance(model, str) else model.name, prompt_sha256=prompt_sha256(think=think))


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


def _messages(pair: dict) -> list[Message]:
    user = USER.format(question=pair['question'], answer=(pair['reference_answer'] or '')[:MAX_ANSWER_CHARS],
                       document_type=pair['document_type'], title=pair.get('title') or '',
                       text=pair['text'][:MAX_CHUNK_CHARS])
    return [Message(role='system', content=SYSTEM), Message(role='user', content=user)]


def judge_pool(pairs: Iterable[dict], model: ChatModel, store: JudgmentStore, *, think: bool,
               progress: Callable[[str], None] | None = None, retry_failed: bool = False) -> dict[str, int]:
    """Judge each unjudged pair, making up to JUDGE_ATTEMPTS tries on schema failures.

    A grade 2 whose evidence quote is not found verbatim in the shown chunk is stored as 1; the
    model's own grade, the quote and the verification result are kept in ``details``.
    When every attempt fails, the pair is stored with grade None, details None and the raw text.
    Transport errors are not caught: the run stops and a rerun skips pairs already in the store.
    With ``retry_failed``, pairs whose latest row is a failure are judged again and a new row is appended.
    """
    model_think = getattr(model, 'think', None)
    if model_think is not None and bool(model_think) != bool(think):
        raise ValueError(f'model think={model_think} but judge_pool think={think}; the key would be wrong')
    key = judge_key(model, think=think)
    max_tokens = generation_settings(think=think)['max_tokens']
    counts = {'judged': 0, 'skipped': 0, 'failed': 0}
    pairs = list(pairs)
    for number, pair in enumerate(pairs, 1):
        if store.has(key, pair['case_id'], pair['chunk_id']) and not (
                retry_failed and store.is_failed(key, pair['case_id'], pair['chunk_id'])):
            counts['skipped'] += 1
            continue
        grade, reason, raw, details = None, '', None, None
        for _attempt in range(JUDGE_ATTEMPTS):
            try:
                parsed = model.generate(_messages(pair), response_schema=RelevanceGrade,
                                        temperature=JUDGE_TEMPERATURE, max_tokens=max_tokens).parsed
            except StructuredOutputError as exc:
                raw = exc.text
                continue
            verified = verify_quote(parsed.evidence_quote, pair['text'][:MAX_CHUNK_CHARS])
            grade = 1 if parsed.grade == 2 and not verified else parsed.grade
            reason = parsed.reason
            details = {'model_grade': parsed.grade, 'evidence_quote': parsed.evidence_quote,
                       'quote_verified': verified}
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
                     think: bool) -> dict:
    return {'finished_at': datetime.now(timezone.utc).isoformat(), 'judge_model': key.model,
            'prompt_sha256': key.prompt_sha256, 'think': think, 'generation': generation_settings(think=think),
            'quote_rule_version': QUOTE_RULE_VERSION, 'model_digest': model_digest,
            'calibration_csv_sha256': calibration_csv_sha256, 'calibration_only': calibration_only,
            'retry_failed': retry_failed, 'pairs': pairs, 'counts': counts}


def append_judge_run(pool_dir: Path, record: dict) -> Path:
    path = Path(pool_dir) / 'judge_runs.jsonl'
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    return path

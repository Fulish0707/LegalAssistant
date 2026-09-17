"""Grade pooled (question, chunk) pairs with a chat model.

The prompt shows the question, the reference answer and the chunk only — never the
retrieval mode, rank, or whether the chunk belongs to the gold document.
"""
from __future__ import annotations

import hashlib
import json
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
JUDGE_MAX_TOKENS = 256
JUDGE_ATTEMPTS = 2
JUDGE_NUM_CTX = 8192
JUDGE_THINK = False

SYSTEM = """당신은 한국 법률 검색 평가자입니다. 질문, 참조 답변, 검색된 문서 조각이 주어집니다.
문서 조각이 참조 답변의 근거로 얼마나 쓸모 있는지 등급을 매기십시오.

2 = 참조 답변이 의존하는 법리·규정·판단 기준을 이 조각이 직접 담고 있어, 이 조각만으로 답변을 뒷받침할 수 있음.
    참조 답변을 작성할 때 쓰인 문서가 아니더라도 같은 법리를 명시하면 2입니다.
1 = 같은 법리나 조문을 다루지만 이 조각만으로는 답변을 뒷받침하기에 부족함.
0 = 무관하거나 용어만 겹침.

JSON으로만 답하십시오: {"grade": 0|1|2, "reason": "한 문장 근거"}"""

USER = """[질문]
{question}

[참조 답변]
{answer}

[문서 조각] ({document_type}) {title}
{text}"""


class RelevanceGrade(BaseModel):
    grade: Literal[0, 1, 2]
    reason: str


def prompt_sha256(*, think: bool = JUDGE_THINK) -> str:
    """Hash of everything that shapes a grade: prompt, schema, truncation and generation settings."""
    return hashlib.sha256(
        json.dumps([SYSTEM, USER, RelevanceGrade.model_json_schema(), MAX_CHUNK_CHARS, MAX_ANSWER_CHARS,
                    JUDGE_TEMPERATURE, JUDGE_MAX_TOKENS, JUDGE_ATTEMPTS, JUDGE_NUM_CTX, think],
                   ensure_ascii=False, sort_keys=True).encode()).hexdigest()


PROMPT_SHA256 = prompt_sha256()


def judge_key(model: ChatModel) -> JudgeKey:
    return JudgeKey(model=model.name, prompt_sha256=PROMPT_SHA256)


def _messages(pair: dict) -> list[Message]:
    user = USER.format(question=pair['question'], answer=(pair['reference_answer'] or '')[:MAX_ANSWER_CHARS],
                       document_type=pair['document_type'], title=pair.get('title') or '',
                       text=pair['text'][:MAX_CHUNK_CHARS])
    return [Message(role='system', content=SYSTEM), Message(role='user', content=user)]


def judge_pool(pairs: Iterable[dict], model: ChatModel, store: JudgmentStore, *,
               progress: Callable[[str], None] | None = None, retry_failed: bool = False) -> dict[str, int]:
    """Judge each unjudged pair, making up to JUDGE_ATTEMPTS tries on schema failures.

    When every attempt fails, the pair is stored with grade None and the raw text. Transport errors
    are not caught: the run stops and a rerun skips pairs already in the store. With ``retry_failed``,
    pairs whose latest row is a failure are judged again and a new row is appended.
    """
    key = judge_key(model)
    counts = {'judged': 0, 'skipped': 0, 'failed': 0}
    pairs = list(pairs)
    for number, pair in enumerate(pairs, 1):
        if store.has(key, pair['case_id'], pair['chunk_id']) and not (
                retry_failed and store.is_failed(key, pair['case_id'], pair['chunk_id'])):
            counts['skipped'] += 1
            continue
        grade, reason, raw = None, '', None
        for _attempt in range(JUDGE_ATTEMPTS):
            try:
                parsed = model.generate(_messages(pair), response_schema=RelevanceGrade,
                                        temperature=JUDGE_TEMPERATURE, max_tokens=JUDGE_MAX_TOKENS).parsed
                grade, reason = parsed.grade, parsed.reason
                break
            except StructuredOutputError as exc:
                raw = exc.text
        store.append(key, case_id=pair['case_id'], chunk_id=pair['chunk_id'], document_id=pair['document_id'],
                     grade=grade, reason=reason, raw=None if grade is not None else raw)
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
                     retry_failed: bool, pairs: int, counts: dict[str, int], model_digest: str | None) -> dict:
    return {'finished_at': datetime.now(timezone.utc).isoformat(), 'judge_model': key.model,
            'prompt_sha256': key.prompt_sha256, 'model_digest': model_digest,
            'calibration_csv_sha256': calibration_csv_sha256, 'calibration_only': calibration_only,
            'retry_failed': retry_failed, 'pairs': pairs, 'counts': counts}


def append_judge_run(pool_dir: Path, record: dict) -> Path:
    path = Path(pool_dir) / 'judge_runs.jsonl'
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    return path

"""Grade pooled (question, chunk) pairs with a chat model.

The prompt shows the question, the reference answer and the chunk only — never the
retrieval mode, rank, or whether the chunk belongs to the gold document.
"""
from __future__ import annotations

import hashlib
import json
from typing import Callable, Iterable, Literal

from pydantic import BaseModel

from legal_rag.llm.base import ChatModel, Message, StructuredOutputError

from .judgment_store import JudgeKey, JudgmentStore

MAX_CHUNK_CHARS = 2500
MAX_ANSWER_CHARS = 800

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


PROMPT_SHA256 = hashlib.sha256(
    json.dumps([SYSTEM, USER, RelevanceGrade.model_json_schema(), MAX_CHUNK_CHARS, MAX_ANSWER_CHARS],
               ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def judge_key(model: ChatModel) -> JudgeKey:
    return JudgeKey(model=model.name, prompt_sha256=PROMPT_SHA256)


def _messages(pair: dict) -> list[Message]:
    user = USER.format(question=pair['question'], answer=(pair['reference_answer'] or '')[:MAX_ANSWER_CHARS],
                       document_type=pair['document_type'], title=pair.get('title') or '',
                       text=pair['text'][:MAX_CHUNK_CHARS])
    return [Message(role='system', content=SYSTEM), Message(role='user', content=user)]


def judge_pool(pairs: Iterable[dict], model: ChatModel, store: JudgmentStore, *,
               progress: Callable[[str], None] | None = None) -> dict[str, int]:
    """Judge each unjudged pair, retrying a schema failure once.

    A second schema failure is stored with grade None and the raw text. Transport errors
    are not caught: the run stops and a rerun skips pairs already in the store.
    """
    key = judge_key(model)
    counts = {'judged': 0, 'skipped': 0, 'failed': 0}
    pairs = list(pairs)
    for number, pair in enumerate(pairs, 1):
        if store.has(key, pair['case_id'], pair['chunk_id']):
            counts['skipped'] += 1
            continue
        grade, reason, raw = None, '', None
        for _attempt in range(2):
            try:
                parsed = model.generate(_messages(pair), response_schema=RelevanceGrade,
                                        temperature=0.0, max_tokens=256).parsed
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

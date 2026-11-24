"""Turn a follow-up into a question that stands alone, so retrieval and reranking can use it.

"그 신청은 어디에 하나요?" retrieved as written returns unrelated chunks (reranker scores 0.01-0.02,
2026-10-05), and so does a user's bare answer to a clarification question. With earlier turns the
question is rewritten once; without them the message is returned untouched and the model is not called.

The rewrite becomes the first search query and the rerank query, so its form is checked in code: the
model sometimes answers instead of asking, or copies a source handle from an earlier answer
(data/eval/multiturn_check, 4 of 13 rewrites). A rewrite that fails the check is replaced by the
user's own words from the window, which are never an answer.
"""

from __future__ import annotations

import re

from legal_rag.llm import ChatModel, Message
from legal_rag.llm.prompts import load_prompt
from legal_rag.schemas import StandaloneQuestion, Turn

from .conversation import SOURCE_HANDLE, render_turns

MAX_QUESTION_CHARS = 1000  # a longer "question" is the model answering or looping, not asking
MAX_SENTENCES = 2  # the prompt asks for one question sentence, or two
_SENTENCE_BREAK = re.compile(r"(?<=[.?!])\s+")
# How a Korean question or request ends; matched on the last sentence once closing marks are removed.
_ASKING_ENDING = re.compile(r"(나요|까요|니까|는지|인지|은지|던지|가요|을까|ㄹ까|세요|십시오|주실래요|궁금합니다|싶습니다|싶어요)$")


def is_standalone_question(text: str) -> bool:
    """True when `text` has the form of a question: no source handle, at most two sentences, and the
    last one asks. Form only; whether it is the right question is not judged here."""
    text = " ".join(text.split())
    if not text or SOURCE_HANDLE.search(text):
        return False
    sentences = [part for part in _SENTENCE_BREAK.split(text) if part]
    if len(sentences) > MAX_SENTENCES:
        return False
    last = sentences[-1]
    if last.endswith("?"):
        return True
    return bool(_ASKING_ENDING.search(last.rstrip(".!\"'”’)」 ")))


class QueryCondenser:
    def __init__(self, model: ChatModel, *, history_turns: int = 4) -> None:
        self.model = model
        self.history_turns = history_turns

    def condense(self, message: str, history: list[Turn]) -> str:
        recent = history[-self.history_turns :] if self.history_turns > 0 else []
        if not recent:
            return message
        prompt = "\n".join(["이전 대화:", *render_turns(recent), "", f"현재 질문: {message}"])
        try:
            response = self.model.generate(
                [Message(role="system", content=load_prompt("condense_system")), Message(role="user", content=prompt)],
                response_schema=StandaloneQuestion,
                temperature=0.0,
            )
        except Exception:  # the unchanged message is always a usable query
            return message
        question = response.parsed.question.strip() if isinstance(response.parsed, StandaloneQuestion) else ""
        if question and len(question) <= MAX_QUESTION_CHARS and is_standalone_question(question):
            return question
        users = [turn.content.strip() for turn in recent if turn.role == "user" and turn.content.strip()]
        return " ".join([*users, message])

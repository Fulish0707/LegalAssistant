"""Turn a follow-up into a question that stands alone, so retrieval and reranking can use it.

"그 신청은 어디에 하나요?" retrieved as written returns unrelated chunks (reranker scores 0.01-0.02,
2026-10-05), and so does a user's bare answer to a clarification question. With earlier turns the
question is rewritten once; without them the message is returned untouched and the model is not called.
"""

from __future__ import annotations

from legal_rag.llm import ChatModel, Message
from legal_rag.llm.prompts import load_prompt
from legal_rag.schemas import StandaloneQuestion, Turn

from .conversation import render_turns

MAX_QUESTION_CHARS = 1000  # a longer "question" is the model answering or looping, not asking


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
        return question if question and len(question) <= MAX_QUESTION_CHARS else message

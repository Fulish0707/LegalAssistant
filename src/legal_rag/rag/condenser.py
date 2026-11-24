"""Turn a follow-up into a question that stands alone, so retrieval and reranking can use it.

"그 신청은 어디에 하나요?" retrieved as written returns unrelated chunks (reranker scores 0.01-0.02,
2026-10-05), and so does a user's bare answer to a clarification question. With earlier turns the
question is rewritten once; without them the message is returned untouched and the model is not called.

The rewrite becomes the first search query and the rerank query, so a rewrite holding a fact the
conversation does not (a date or amount the user never gave, an article no turn named) is not used:
the user's own words from the window are joined instead (data/eval/condense_facts).

An earlier answer reaches the condenser as its first line, the summary (condenser_view): given the
whole analysis, the model continued it and wrote an answer instead of a question
(data/eval/condense_summary).
"""

from __future__ import annotations

from legal_rag.llm import ChatModel, Message
from legal_rag.llm.prompts import load_prompt
from legal_rag.schemas import StandaloneQuestion, Turn
from legal_rag.textutil.refs import extract_case_refs, extract_statute_refs
from legal_rag.verification.quantities import extract_quantities

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
        prompt = "\n".join(["이전 대화:", *render_turns(condenser_view(recent)), "", f"현재 질문: {message}"])
        try:
            response = self.model.generate(
                [Message(role="system", content=load_prompt("condense_system")), Message(role="user", content=prompt)],
                response_schema=StandaloneQuestion,
                temperature=0.0,
            )
        except Exception:  # the unchanged message is always a usable query
            return message
        question = response.parsed.question.strip() if isinstance(response.parsed, StandaloneQuestion) else ""
        if not question or len(question) > MAX_QUESTION_CHARS:
            return message
        users = [turn.content for turn in recent if turn.role == "user"] + [message]
        window = [turn.content for turn in recent] + [message]
        if invented_facts(question, users=users, window=window):
            return " ".join(text.strip() for text in users if text.strip())
        return question


def condenser_view(turns: list[Turn]) -> list[Turn]:
    """The turns as the condenser reads them: an answer turn (an assistant turn that cites evidence,
    stored as summary, newline, analysis) is cut to its first line, the summary. A clarification or
    an explanation (no evidence) and the user's turns are kept whole."""
    return [turn.model_copy(update={"content": turn.content.split("\n", 1)[0]})
            if turn.role == "assistant" and turn.evidence_ids else turn for turn in turns]


def invented_facts(question: str, *, users: list[str], window: list[str]) -> list[str]:
    """Facts in `question` the conversation does not hold: a date, period, amount, count or rate the
    user never gave (the model writes the prompt's example date, or "2023년 3월에 500만원" for a loan
    of no stated amount), or an article or case number in no turn of the window."""
    user_text = "\n".join(users)
    # Read the user leniently, so the check only misses: "석 달째" is a duration here, not an ordinal,
    # and a quantity written exactly as the user wrote it ("10월 1일" of "2025년 10월 1일") is theirs.
    user_keys = {q.key for q in extract_quantities(user_text.replace("째", ""))}
    user_squashed = "".join(user_text.split())
    invented = [q.value for q in extract_quantities(question)
                if q.key not in user_keys and "".join(question[q.start:q.end].split()) not in user_squashed]
    window_text = "\n".join(window)
    articles = {ref.article for ref in extract_statute_refs(window_text)}
    invented += [ref.render() for ref in extract_statute_refs(question) if ref.article not in articles]
    cases = set(extract_case_refs(window_text))
    invented += [case for case in extract_case_refs(question) if case not in cases]
    return invented

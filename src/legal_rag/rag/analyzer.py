"""Query understanding: deterministic reference extraction plus an LLM structured pass."""

from __future__ import annotations

from legal_rag.llm import ChatModel, Message
from legal_rag.llm.prompts import load_prompt
from legal_rag.schemas import QueryAnalysis, Turn
from legal_rag.textutil.refs import extract_case_refs, extract_statute_refs


class QueryAnalyzer:
    def __init__(self, model: ChatModel, *, history_turns: int = 4) -> None:
        self.model = model
        self.history_turns = history_turns

    def analyze(self, message: str, history: list[Turn]) -> QueryAnalysis:
        statute_refs = [ref.render() for ref in extract_statute_refs(message)]
        case_refs = extract_case_refs(message)
        recent = history[-self.history_turns :] if self.history_turns > 0 else []
        try:
            response = self.model.generate(
                [Message(role="system", content=load_prompt("analyze_system")), Message(role="user", content=self._render(message, recent))],
                response_schema=QueryAnalysis,
                temperature=0.0,
            )
            base = response.parsed if isinstance(response.parsed, QueryAnalysis) else QueryAnalysis()
        except Exception:  # the deterministic pass is always available
            base = QueryAnalysis()
        return base.model_copy(update={"statute_refs": statute_refs or base.statute_refs, "case_refs": case_refs or base.case_refs})

    @staticmethod
    def _render(message: str, history: list[Turn]) -> str:
        lines: list[str] = []
        if history:
            lines.append("이전 대화:")
            lines += [f"- {'사용자' if turn.role == 'user' else '상담'}: {turn.content}" for turn in history]
            lines.append("")
        lines.append(f"현재 질문: {message}")
        return "\n".join(lines)

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
        # The model sees earlier answers and copies their articles and case numbers. A reference
        # counts as the user's (a search query, and exempt from the citation check) only when
        # the user's own messages contain it.
        written = "\n".join([message, *(turn.content for turn in recent if turn.role == "user")])
        articles = {ref.article for ref in extract_statute_refs(written)}
        cases = set(extract_case_refs(written))
        squashed = "".join(written.split())
        def verbatim(ref):  # a law named without an article
            return "".join(ref.split()) in squashed
        model_statutes = [ref for ref in base.statute_refs if verbatim(ref) or any(found.article in articles for found in extract_statute_refs(ref))]
        model_cases = [ref for ref in base.case_refs if verbatim(ref) or any(found in cases for found in extract_case_refs(ref))]
        return base.model_copy(update={"statute_refs": statute_refs or model_statutes, "case_refs": case_refs or model_cases})

    @staticmethod
    def _render(message: str, history: list[Turn]) -> str:
        lines: list[str] = []
        if history:
            lines.append("이전 대화:")
            lines += [f"- {'사용자' if turn.role == 'user' else '상담'}: {turn.content}" for turn in history]
            lines.append("")
        lines.append(f"현재 질문: {message}")
        return "\n".join(lines)

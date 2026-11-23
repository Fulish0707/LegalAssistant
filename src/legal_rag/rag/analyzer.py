"""Query understanding: deterministic reference extraction plus an LLM structured pass."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from legal_rag.llm import ChatModel, Message
from legal_rag.llm.prompts import load_prompt
from legal_rag.schemas import QueryAnalysis, Turn
from legal_rag.textutil.refs import extract_case_refs, extract_statute_refs

if TYPE_CHECKING:
    from legal_rag.config import Settings


class QueryAnalysisCore(BaseModel):
    """The QueryAnalysis fields the pipeline reads when analyzer terms and the hard filter are off.
    Asking for the full schema made EXAONE 3.5 write 200-250 tokens, 5-6 s per turn (2026-11-23)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    domain: str | None = None
    needs_more_info: bool = False
    missing_facts: list[str] = Field(default_factory=list)
    statute_refs: list[str] = Field(default_factory=list)
    case_refs: list[str] = Field(default_factory=list)


def needs_full_analysis(settings: Settings) -> bool:
    """issues/keywords feed the search only with analyzer_terms_in_search; document_types_needed only with hard_filter."""
    return bool(settings.rag.analyzer_terms_in_search or settings.rag.hard_filter)


class QueryAnalyzer:
    def __init__(self, model: ChatModel, *, history_turns: int = 4, full: bool = True) -> None:
        self.model = model
        self.history_turns = history_turns
        self.full = full

    def analyze(self, message: str, history: list[Turn]) -> QueryAnalysis:
        statute_refs = [ref.render() for ref in extract_statute_refs(message)]
        case_refs = extract_case_refs(message)
        recent = history[-self.history_turns :] if self.history_turns > 0 else []
        schema, prompt = (QueryAnalysis, "analyze_system") if self.full else (QueryAnalysisCore, "analyze_core_system")
        try:
            response = self.model.generate(
                [Message(role="system", content=load_prompt(prompt)), Message(role="user", content=self._render(message, recent))],
                response_schema=schema,
                temperature=0.0,
            )
            parsed = response.parsed
            base = QueryAnalysis.model_validate(parsed.model_dump()) if isinstance(parsed, schema) else QueryAnalysis()
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

"""Turn one question into a few legal-search phrasings, and derive retrieval filters."""

from __future__ import annotations

from legal_rag.llm import ChatModel, Message
from legal_rag.llm.prompts import load_prompt
from legal_rag.schemas import QueryAnalysis, RetrievalFilters, RewrittenQueries, citation_kind


class QueryRewriter:
    def __init__(self, model: ChatModel | None = None, *, max_queries: int = 4) -> None:
        self.model = model
        self.max_queries = max_queries

    def rewrite(self, message: str, analysis: QueryAnalysis) -> RewrittenQueries:
        queries = [message, *analysis.statute_refs, *analysis.case_refs]
        terms = list(dict.fromkeys([*analysis.issues, *analysis.keywords]))
        if terms:
            queries.append(" ".join(terms))
        if self.model is not None:
            try:
                response = self.model.generate(
                    [Message(role="system", content=load_prompt("rewrite_system")), Message(role="user", content=message)],
                    response_schema=RewrittenQueries,
                    temperature=0.0,
                )
                if isinstance(response.parsed, RewrittenQueries):
                    queries += response.parsed.search_queries
            except Exception:
                pass
        deduped = list(dict.fromkeys(query.strip() for query in queries if query and query.strip()))
        return RewrittenQueries(original=message, search_queries=deduped[: self.max_queries])

    @staticmethod
    def filters(analysis: QueryAnalysis) -> RetrievalFilters | None:
        # Only a single, certain kind of authority becomes a hard filter. Legal domain stays a
        # soft signal: statutes rarely carry a domain label, so filtering on it would hide them.
        # Filter by citation kind, never by exact document_type: the analyzer names the native
        # court_decision type, which no imported corpus uses (cases are judgment, decision, ...),
        # so an exact filter matched nothing and the answer had no evidence (2026-09-27).
        kinds = {citation_kind(document_type) for document_type in analysis.document_types_needed}
        if len(kinds) != 1:
            return None
        return RetrievalFilters(citation_kind=kinds.pop())

"""Query understanding contracts."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from .answer import CitationKind
from .documents import DocumentType


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class QueryAnalysis(_Frozen):
    """Structured reading of the user's question. Every field is optional for the LLM."""

    domain: str | None = None
    subdomain: str | None = None
    issues: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    statute_refs: list[str] = Field(default_factory=list)
    case_refs: list[str] = Field(default_factory=list)
    document_types_needed: list[DocumentType] = Field(default_factory=list)
    needs_more_info: bool = False
    missing_facts: list[str] = Field(default_factory=list)


class RetrievalFilters(_Frozen):
    document_types: list[DocumentType] | None = None
    citation_kind: CitationKind | None = None  # coarse facet; expands to document_types
    legal_domain: str | None = None
    court: str | None = None
    law_name: str | None = None
    decided_after: date | None = None
    decided_before: date | None = None
    effective_on: date | None = None

    def is_empty(self) -> bool:
        return self == RetrievalFilters()


class RewrittenQueries(_Frozen):
    original: str
    search_queries: list[str] = Field(default_factory=list)  # original always first

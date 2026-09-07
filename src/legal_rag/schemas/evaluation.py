"""Evaluation dataset rows, one JSON object per line."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .query import RetrievalFilters


class RetrievalExample(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    query: str
    relevant_document_ids: list[str]
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    filters: RetrievalFilters | None = None


class GenerationExample(BaseModel):
    """Used from V0.4 onward; defined now so datasets can be authored early."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    query: str
    history: list[str] = Field(default_factory=list)
    expected_issues: list[str] = Field(default_factory=list)
    expected_statutes: list[str] = Field(default_factory=list)
    expected_cases: list[str] = Field(default_factory=list)
    expect_clarification: bool = False
    forbidden_references: list[str] = Field(default_factory=list)

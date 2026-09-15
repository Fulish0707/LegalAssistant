"""Retrieval units derived from canonical documents."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from .documents import DocumentType


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ChunkMetadata(_Frozen):
    original: dict[str, JsonValue] = Field(default_factory=dict)
    dataset: str
    corpus_version: str
    law_name: str | None = None
    article: str | None = None
    paragraph: str | None = None
    court: str | None = None
    case_number: str | None = None
    decision_date: date | None = None
    authority: str | None = None
    reference_number: str | None = None
    legal_domain: str | None = None
    effective_date: date | None = None
    section: str | None = None


class Chunk(_Frozen):
    id: str  # "{document_id}#{path}"
    document_id: str
    document_type: DocumentType
    path: str  # "art:제750조", "art:제3조의2/para:제1항", "sec:reasoning/2"
    order: int
    text: str  # retrieval text; first line is a structural header
    title: str
    metadata: ChunkMetadata


class ScoredChunk(_Frozen):
    chunk: Chunk
    score: float
    retriever: str  # "dense" | "sparse" | "fused" | "reranked"

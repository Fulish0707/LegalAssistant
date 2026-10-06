"""Request and response bodies for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from legal_rag.schemas import RetrievalFilters, ScoredChunk, UserDocument


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=128)
    include_trace: bool = False
    # Passages of the caller's own files that bear on this message; the service holds no user files.
    user_documents: list[UserDocument] = Field(default_factory=list, max_length=3)


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=50)
    filters: RetrievalFilters | None = None


class SearchResponse(BaseModel):
    hits: list[ScoredChunk]
    # document_id -> display label, for hits whose chunk names no court, case, law or title.
    labels: dict[str, str] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    corpus_version: str
    collection: str | None = None
    chunks: int | None = None
    embedder: str | None = None
    llm: str
    tokenizer: str | None = None
    warm: bool = False  # the startup warm-up search has completed

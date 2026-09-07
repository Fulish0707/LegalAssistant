"""Request and response bodies for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from legal_rag.schemas import RetrievalFilters, ScoredChunk


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=128)
    include_trace: bool = False


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=50)
    filters: RetrievalFilters | None = None


class SearchResponse(BaseModel):
    hits: list[ScoredChunk]


class HealthResponse(BaseModel):
    status: str
    corpus_version: str
    collection: str | None = None
    chunks: int | None = None
    embedder: str | None = None
    llm: str
    tokenizer: str | None = None

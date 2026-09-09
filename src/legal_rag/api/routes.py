"""HTTP routes. Business logic lives in the orchestrator and retriever, never here."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool

from legal_rag.schemas import ChatResponse

from .models import ChatRequest, HealthResponse, SearchRequest, SearchResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    state = request.app.state
    settings = state.settings
    metadata = state.index_metadata() or {}
    return HealthResponse(
        status="ok",
        corpus_version=settings.corpus.version,
        collection=metadata.get("collection"),
        chunks=metadata.get("chunks"),
        embedder=metadata.get("embedder"),
        llm=settings.llm.backend if settings.llm.backend == "fake" else f"{settings.llm.backend}:{settings.llm.model}",
        tokenizer=metadata.get("tokenizer"),
    )


@router.post("/v1/search", response_model=SearchResponse)
async def search(body: SearchRequest, request: Request) -> SearchResponse:
    retriever = request.app.state.get_retriever()
    hits = await run_in_threadpool(retriever.retrieve, body.query, body.k, body.filters)
    return SearchResponse(hits=hits)


@router.post("/v1/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    orchestrator = request.app.state.get_orchestrator()
    conversation_id = body.conversation_id or uuid.uuid4().hex
    response = await run_in_threadpool(orchestrator.handle, body.message, conversation_id=conversation_id)
    if not body.include_trace:
        response = response.model_copy(update={"trace": None})
    return response

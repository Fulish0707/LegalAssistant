"""HTTP routes for the authenticated internal service."""

from __future__ import annotations

import uuid
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from legal_rag.retrieval.labels import document_labels
from legal_rag.schemas import ChatResponse

from .models import ChatRequest, HealthResponse, SearchRequest, SearchResponse
from .security import check_egress, conversation_key, workspace

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


async def _run(request, operation):
    # A cancelled HTTP request does not release a slot while its worker is still running.
    def guarded():
        with request.app.state.security.work():
            try:
                return operation()
            except HTTPException:
                raise
            except Exception as error:
                logger.error("Internal request failed (%s)", type(error).__name__)
                raise HTTPException(500, "service_processing_failed") from None
    return await run_in_threadpool(guarded)


@router.post("/v1/search", response_model=SearchResponse)
async def search(body: SearchRequest, request: Request, owner: str = Depends(workspace)) -> SearchResponse:
    check_egress(request.app.state.settings)
    def retrieve():
        retriever = request.app.state.get_retriever()
        hits = retriever.retrieve(body.query, body.k, body.filters)
        stores = {leg.name.rsplit(":", 1)[0]: leg.store for leg in getattr(retriever, "legs", []) if hasattr(leg, "store")}
        return SearchResponse(hits=hits, labels=document_labels(hits, stores))
    return await _run(request, retrieve)


@router.post("/v1/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat(body: ChatRequest, request: Request, owner: str = Depends(workspace)) -> ChatResponse:
    settings = request.app.state.settings
    if settings.trace.include_text and not settings.api.allow_trace:
        raise HTTPException(503, "text_trace_disabled")
    check_egress(settings, documents=bool(body.user_documents))
    if body.include_trace and not settings.api.allow_trace:
        raise HTTPException(403, "trace_access_disabled")
    external_id = body.conversation_id or uuid.uuid4().hex
    def handle():
        orchestrator = request.app.state.get_orchestrator()
        if not orchestrator.store.has_capacity(owner + ":"):
            raise HTTPException(429, "workspace_storage_limit")
        response = orchestrator.handle(body.message, conversation_id=conversation_key(owner, external_id),
                                       user_documents=body.user_documents)
        return response.model_copy(update={"conversation_id": external_id,
                                           "trace": response.trace if body.include_trace else None})
    return await _run(request, handle)


@router.delete("/v1/workspace/conversations")
async def delete_conversations(request: Request, owner: str = Depends(workspace)) -> dict:
    def clear():
        orchestrator = request.app.state.get_orchestrator()
        orchestrator.store.delete_prefix(owner + ":")
        return {"ok": True}
    return await _run(request, clear)

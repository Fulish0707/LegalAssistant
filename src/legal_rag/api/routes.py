"""HTTP routes. Business logic lives in the orchestrator and retriever, never here."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool

from legal_rag.retrieval.document import find_document
from legal_rag.retrieval.labels import document_labels
from legal_rag.schemas import ChatResponse

from .models import ChatRequest, DocumentResponse, HealthResponse, SearchRequest, SearchResponse

router = APIRouter()


def _stores(retriever) -> dict:
    """One vector store per corpus the retriever searches."""
    return {leg.name.rsplit(":", 1)[0]: leg.store for leg in getattr(retriever, "legs", []) if hasattr(leg, "store")}


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
        warm=state.warm,
    )


@router.post("/v1/search", response_model=SearchResponse)
async def search(body: SearchRequest, request: Request) -> SearchResponse:
    retriever = request.app.state.get_retriever()
    hits = await run_in_threadpool(retriever.retrieve, body.query, body.k, body.filters)
    labels = await run_in_threadpool(document_labels, hits, _stores(retriever))
    return SearchResponse(hits=hits, labels=labels)


@router.get("/v1/document", response_model=DocumentResponse)
async def document(request: Request, chunk_id: str = Query(min_length=1, max_length=512)) -> DocumentResponse:
    """The whole document that holds one chunk, for reading a source an answer cited."""
    stores = list(_stores(request.app.state.get_retriever()).values())
    found = await run_in_threadpool(find_document, stores, chunk_id)
    if found is None:
        raise HTTPException(status_code=404, detail="No indexed document holds this chunk.")
    chunk, text, start, end = found
    return DocumentResponse(document_id=chunk.document_id, text=text, cited_start=start, cited_end=end)


@router.post("/v1/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    orchestrator = request.app.state.get_orchestrator()
    conversation_id = body.conversation_id or uuid.uuid4().hex
    response = await run_in_threadpool(orchestrator.handle, body.message, conversation_id=conversation_id,
                                         user_documents=body.user_documents)
    if not body.include_trace:
        response = response.model_copy(update={"trace": None})
    return response

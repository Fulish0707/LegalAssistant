"""HTTP routes for the authenticated internal service."""

from __future__ import annotations

import json
import logging
import queue
import threading
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from legal_rag.retrieval.document import find_document, pinned_chunks
from legal_rag.retrieval.labels import document_labels
from legal_rag.retrieval.laws import articles, compact_names, statutes_cited
from legal_rag.schemas import ChatResponse, CitationKind, RetrievalFilters

from .models import (CasesRequest, CasesResponse, ChatRequest, CitedArticle, DocumentResponse, HealthResponse,
                     LawResponse, SearchRequest, SearchResponse)
from .security import check_egress, conversation_key, workspace

router = APIRouter()
logger = logging.getLogger(__name__)


def _stores(retriever) -> dict:
    """One vector store per corpus the retriever searches."""
    return {leg.name.rsplit(":", 1)[0]: leg.store for leg in getattr(retriever, "legs", []) if hasattr(leg, "store")}


def _laws(state) -> dict:
    """Law name -> the store that holds it article by article. Listed once: it changes only with the index."""
    if getattr(state, "laws", None) is None:
        stores, chosen = _stores(state.get_retriever()), state.settings.retrieval.law_corpus
        state.laws = {name: store for version, store in stores.items() if chosen in (None, version)
                      for name in store.law_names()}
    return state.laws


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
    # The getters wait on a lock while the retriever is built (~10 s at start); off the event loop.
    retriever = await run_in_threadpool(request.app.state.get_retriever)
    hits = await run_in_threadpool(retriever.retrieve, body.query, body.k, body.filters)
    labels = await run_in_threadpool(document_labels, hits, _stores(retriever))
    return SearchResponse(hits=hits, labels=labels)


@router.get("/v1/document", response_model=DocumentResponse)
async def document(request: Request, chunk_id: str = Query(min_length=1, max_length=512),
                   owner: str = Depends(workspace)) -> DocumentResponse:
    """The whole document that holds one chunk, for reading a source an answer cited."""
    stores = list(_stores(await run_in_threadpool(request.app.state.get_retriever)).values())
    found = await run_in_threadpool(find_document, stores, chunk_id)
    if found is None:
        raise HTTPException(status_code=404, detail="No indexed document holds this chunk.")
    chunk, text, start, end = found
    return DocumentResponse(document_id=chunk.document_id, text=text, cited_start=start, cited_end=end)


def _cases(state, query: str, k: int) -> CasesResponse:
    retriever = state.get_retriever()
    best = {}  # several passages of one case may rank; the page lists a case once
    for hit in retriever.retrieve(query, k, RetrievalFilters(citation_kind=CitationKind.case)):
        best.setdefault((hit.chunk.metadata.corpus_version, hit.chunk.document_id), hit)
    hits, stores = list(best.values()), _stores(retriever)
    cited = statutes_cited(hits, stores, compact_names(_laws(state)))
    return CasesResponse(hits=hits, labels=document_labels(hits, stores), statutes={
        document: [CitedArticle(law=ref.law_name, article=ref.article, paragraph=ref.paragraph) for ref in refs]
        for document, refs in cited.items()})


@router.post("/v1/cases", response_model=CasesResponse)
async def cases(body: CasesRequest, request: Request, owner: str = Depends(workspace)) -> CasesResponse:
    """Cases closest to the query, each with the statute articles its text cites."""
    return await run_in_threadpool(_cases, request.app.state, body.query, body.k)


@router.get("/v1/law", response_model=LawResponse)
async def law(request: Request, name: str = Query(min_length=1, max_length=200),
              owner: str = Depends(workspace)) -> LawResponse:
    """One law's articles in order, each with its heading and full text."""
    store = (await run_in_threadpool(_laws, request.app.state)).get(name)
    if store is None:
        raise HTTPException(status_code=404, detail="No indexed law has this name.")
    return LawResponse(name=name, articles=articles(await run_in_threadpool(store.law_chunks, name)))


def _check_chat(body: ChatRequest, settings) -> None:
    if settings.trace.include_text and not settings.api.allow_trace:
        raise HTTPException(503, "text_trace_disabled")
    check_egress(settings, documents=bool(body.user_documents))
    if body.include_trace and not settings.api.allow_trace:
        raise HTTPException(403, "trace_access_disabled")


async def _turn(body: ChatRequest, request: Request, owner: str):
    """The orchestrator, the caller's conversation id, its store key and the pinned passages for one turn."""
    _check_chat(body, request.app.state.settings)
    orchestrator = await run_in_threadpool(request.app.state.get_orchestrator)
    if not await run_in_threadpool(orchestrator.store.has_capacity, owner + ":"):
        raise HTTPException(429, "workspace_storage_limit")
    pinned = []
    if body.pinned_chunk_ids:
        stores = list(_stores(await run_in_threadpool(request.app.state.get_retriever)).values())
        pinned = await run_in_threadpool(pinned_chunks, stores, body.pinned_chunk_ids)
    conversation_id = body.conversation_id or uuid.uuid4().hex
    return orchestrator, conversation_id, conversation_key(owner, conversation_id), pinned


@router.post("/v1/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat(body: ChatRequest, request: Request, owner: str = Depends(workspace)) -> ChatResponse:
    orchestrator, conversation_id, key, pinned = await _turn(body, request, owner)
    response = await _run(request, lambda: orchestrator.handle(body.message, conversation_id=key,
                                                               user_documents=body.user_documents, pinned=pinned))
    # The caller sees its own id; the store keys the turns by workspace (conversation_key).
    return response.model_copy(update={"conversation_id": conversation_id,
                                       "trace": response.trace if body.include_trace else None})


@router.post("/v1/chat/stream")
async def chat_stream(body: ChatRequest, request: Request, owner: str = Depends(workspace)) -> StreamingResponse:
    """/v1/chat as NDJSON: {"type": "stage", "stage"} as each stage starts (see Orchestrator.handle),
    {"type": "draft", "text"} pieces of the analysis while the model writes it, {"type": "draft_reset"}
    before each new attempt, then {"type": "response", "body"} with the /v1/chat body, or
    {"type": "error"}. The draft has not been verified; only the body has."""
    orchestrator, conversation_id, key, pinned = await _turn(body, request, owner)
    slots = request.app.state.security.slots
    if not slots.acquire(blocking=False):
        raise HTTPException(429, "service_busy")
    events: queue.Queue = queue.Queue()

    def draft(text):
        events.put({"type": "draft_reset"} if text is None else {"type": "draft", "text": text})

    def run():
        # The slot is held until the turn ends, also when the client has gone: the turn still writes history.
        try:
            response = orchestrator.handle(body.message, conversation_id=key, user_documents=body.user_documents,
                                           pinned=pinned, on_draft=draft,
                                           on_stage=lambda stage: events.put({"type": "stage", "stage": stage}))
            response = response.model_copy(update={"conversation_id": conversation_id,
                                                   "trace": response.trace if body.include_trace else None})
            events.put({"type": "response", "body": response.model_dump(mode="json", exclude_none=True)})
        except Exception as error:
            logger.error("chat turn failed (%s)", type(error).__name__)
            events.put({"type": "error"})
        finally:
            slots.release()
            events.put(None)

    threading.Thread(target=run, daemon=True).start()

    def lines():
        # Starlette runs this iterator in its thread pool, so the wait never blocks the event loop.
        while (event := events.get()) is not None:
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson")


@router.delete("/v1/workspace/conversations")
async def delete_conversations(request: Request, owner: str = Depends(workspace),
                               conversation_id: str | None = Query(default=None, min_length=1, max_length=128,
                                                                   pattern=r"^[A-Za-z0-9_-]+$")) -> dict:
    """Forget the calling workspace's turns: of one conversation, or of all of them."""
    # A key is the owner, ":" and a fixed-length hash, so as a prefix it matches only itself.
    prefix = conversation_key(owner, conversation_id) if conversation_id else owner + ":"

    def clear():
        request.app.state.get_orchestrator().store.delete_prefix(prefix)
        return {"ok": True}
    return await _run(request, clear)

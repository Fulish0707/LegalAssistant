"""HTTP routes. Business logic lives in the orchestrator and retriever, never here."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool

from legal_rag.retrieval.document import find_document
from legal_rag.retrieval.labels import document_labels
from legal_rag.retrieval.laws import articles, compact_names, statutes_cited
from legal_rag.schemas import ChatResponse, CitationKind, RetrievalFilters

from .models import (CasesRequest, CasesResponse, ChatRequest, CitedArticle, DocumentResponse, HealthResponse,
                     LawResponse, SearchRequest, SearchResponse)

router = APIRouter()


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
async def cases(body: CasesRequest, request: Request) -> CasesResponse:
    """Cases closest to the query, each with the statute articles its text cites."""
    return await run_in_threadpool(_cases, request.app.state, body.query, body.k)


@router.get("/v1/law", response_model=LawResponse)
async def law(request: Request, name: str = Query(min_length=1, max_length=200)) -> LawResponse:
    """One law's articles in order, each with its heading and full text."""
    store = (await run_in_threadpool(_laws, request.app.state)).get(name)
    if store is None:
        raise HTTPException(status_code=404, detail="No indexed law has this name.")
    return LawResponse(name=name, articles=articles(await run_in_threadpool(store.law_chunks, name)))


@router.post("/v1/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    orchestrator = request.app.state.get_orchestrator()
    conversation_id = body.conversation_id or uuid.uuid4().hex
    response = await run_in_threadpool(orchestrator.handle, body.message, conversation_id=conversation_id,
                                         user_documents=body.user_documents)
    if not body.include_trace:
        response = response.model_copy(update={"trace": None})
    return response

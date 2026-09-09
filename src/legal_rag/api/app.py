"""FastAPI application factory. Backends are built lazily on first use so tests can inject fakes."""

from __future__ import annotations

import json
from threading import Lock

from fastapi import FastAPI

from legal_rag import __version__
from legal_rag.config import Settings, load_settings
from legal_rag.factory import build_hybrid_retriever, build_orchestrator, index_metadata_path

from .routes import router


def create_app(settings: Settings | None = None, orchestrator=None, retriever=None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="Korean Legal RAG", version=__version__)
    state = app.state
    state.settings = settings
    state.retriever = retriever
    state.orchestrator = orchestrator
    lock = Lock()

    def get_retriever():
        with lock:
            if state.retriever is None:
                state.retriever = build_hybrid_retriever(settings)
            return state.retriever

    def get_orchestrator():
        with lock:
            if state.orchestrator is None:
                state.orchestrator = build_orchestrator(settings, retriever=get_retriever_unlocked())
            return state.orchestrator

    def get_retriever_unlocked():
        if state.retriever is None:
            state.retriever = build_hybrid_retriever(settings)
        return state.retriever

    def index_metadata() -> dict | None:
        path = index_metadata_path(settings)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    state.get_retriever = get_retriever
    state.get_orchestrator = get_orchestrator
    state.index_metadata = index_metadata
    app.include_router(router)
    return app


def app_for_uvicorn() -> FastAPI:
    return create_app()

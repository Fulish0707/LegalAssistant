"""Warm the search stack when the API starts, not on the first user's request.

Cold, the first /v1/search took ~18 s (BM25 load, bge-m3, reranker load); warm, 2.2 s.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from legal_rag.api import create_app
from legal_rag.config import Settings


class _FakeRetriever:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[str] = []
        self.fail = fail

    def retrieve(self, query, k, filters=None):
        self.calls.append(query)
        if self.fail:
            raise RuntimeError("reranker failed to load")
        return []


def _run(settings: Settings, retriever: _FakeRetriever) -> dict:
    app = create_app(settings=settings, retriever=retriever)
    with TestClient(app) as client:
        app.state.warmup_thread and app.state.warmup_thread.join(timeout=5)
        return client.get("/health").json()


def test_warmup_runs_one_search_at_startup_and_health_reports_it():
    retriever = _FakeRetriever()
    body = _run(Settings(api={"warmup": True}), retriever)
    assert len(retriever.calls) == 1
    assert body["warm"] is True


def test_no_warmup_by_default():
    retriever = _FakeRetriever()
    body = _run(Settings(), retriever)
    assert retriever.calls == []
    assert body["warm"] is False


def test_failed_warmup_leaves_the_api_serving():
    retriever = _FakeRetriever(fail=True)
    body = _run(Settings(api={"warmup": True}), retriever)
    assert body["status"] == "ok"
    assert body["warm"] is False

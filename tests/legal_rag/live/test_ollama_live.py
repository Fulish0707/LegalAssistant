"""Live checks against a local Ollama. Skipped unless LEGAL_RAG_LIVE=1.

    LEGAL_RAG_LIVE=1 uv run pytest tests/legal_rag/live -q
Set LEGAL_RAG__EMBEDDING__MODEL to pick an installed embedding model (default bge-m3).
"""

import os
from pathlib import Path

import pytest

from legal_rag.config import load_settings
from legal_rag.factory import build_embedder, build_orchestrator
from legal_rag.ingestion.pipeline import run_ingestion
from legal_rag.retrieval.index_builder import build_index

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "raw"

pytestmark = pytest.mark.skipif(os.getenv("LEGAL_RAG_LIVE") != "1", reason="set LEGAL_RAG_LIVE=1 to run against Ollama")


@pytest.fixture(scope="module")
def live_settings(tmp_path_factory):
    root = tmp_path_factory.mktemp("live")
    return load_settings(profile="default", overrides={"paths": {"root": str(root), "raw_dir": str(FIXTURES)}, "corpus": {"version": "live"}})


def test_embedder_returns_vectors(live_settings):
    embedder = build_embedder(live_settings)
    vector = embedder.embed_query("보증금 반환")
    assert embedder.dimension == len(vector) > 0


def test_grounded_chat_turn_verifies(live_settings):
    run_ingestion(live_settings)
    build_index(live_settings)
    orchestrator = build_orchestrator(live_settings)
    response = orchestrator.handle("집주인이 보증금을 돌려주지 않습니다. 어떻게 해야 하나요?", conversation_id="live")
    assert response.kind in {"answer", "clarification"}
    if response.kind == "answer":
        assert response.trace.policy_action in {"accept", "strip"}
        assert response.evidence and response.evidence.items[0].source_id == "S1"

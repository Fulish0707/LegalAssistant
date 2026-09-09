from pathlib import Path

import pytest

from legal_rag.config import load_settings
from legal_rag.factory import build_hybrid_retriever
from legal_rag.ingestion.pipeline import run_ingestion
from legal_rag.retrieval.index_builder import build_index
from legal_rag.schemas import DocumentType, RetrievalFilters

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "raw"


def fixture_settings(root: Path, corpus_version: str):
    return load_settings(profile="default", overrides={
        "paths": {"root": str(root), "raw_dir": str(FIXTURES)},
        "corpus": {"version": corpus_version},
        "embedding": {"backend": "fake"},
        "retrieval": {"k_final": 3},
    })


@pytest.fixture(scope="module")
def indexed(tmp_path_factory):
    """Ingest + index the fixtures once, and open the single retriever the module shares.

    Embedded Qdrant allows one client per storage path (design risk R6), so every test
    that needs retrieval must reuse this retriever instead of opening another.
    """
    root = tmp_path_factory.mktemp("idx")
    settings = fixture_settings(root, "it")
    run_ingestion(settings)
    report = build_index(settings)
    retriever = build_hybrid_retriever(settings)
    yield settings, report, retriever
    for leg in retriever.legs:
        leg.store.close()
        break


def test_index_report(indexed):
    settings, report, _ = indexed
    assert report.chunks > 5 and report.collection == "legal_it_fake_64"
    assert (settings.index_corpus_dir() / "bm25.json").exists() and (settings.index_corpus_dir() / "index.json").exists()


def test_hybrid_search_finds_the_right_units(indexed):
    _, _, retriever = indexed
    hits = retriever.retrieve("임차권등기명령 신청서에 적어야 하는 사항", k=3)
    assert any(h.chunk.path == "art:제3조의3/para:제2항" for h in hits)
    hits = retriever.retrieve("민법 제750조", k=3)
    assert hits[0].chunk.metadata.article == "제750조" and hits[0].chunk.document_type == DocumentType.statute
    only_cases = retriever.retrieve("보증금 반환", k=3, filters=RetrievalFilters(document_types=[DocumentType.court_decision]))
    assert only_cases and all(h.chunk.document_type == DocumentType.court_decision for h in only_cases)

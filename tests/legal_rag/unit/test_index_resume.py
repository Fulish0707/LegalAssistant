"""build_index --resume: an interrupted run continues without re-embedding or duplicating chunks."""

import json
from unittest.mock import Mock

import httpx
import pytest

from legal_rag.config import Settings
from legal_rag.embeddings import FakeEmbedder
from legal_rag.embeddings.ollama import OllamaEmbedder
from legal_rag.factory import open_store
from legal_rag.retrieval.index_builder import build_index
from legal_rag.retrieval.store import VectorStore
from legal_rag.schemas import Chunk, ChunkMetadata

N = 12


def _write_corpus(settings: Settings, texts: list[str]) -> None:
    directory = settings.processed_corpus_dir()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "documents.jsonl").write_text("")
    chunks = [Chunk(id=f"doc#{i}", document_id="doc", document_type="statute", path=str(i), order=i, text=text,
                    title="law", metadata=ChunkMetadata(dataset="test", corpus_version="v1"))
              for i, text in enumerate(texts)]
    (directory / "chunks.jsonl").write_text("\n".join(chunk.model_dump_json() for chunk in chunks))


@pytest.fixture
def settings(tmp_path):
    # batch_size 1 -> the first build uploads 4 chunks per batch, so it can fail between batches.
    settings = Settings(paths={"root": tmp_path}, embedding={"backend": "fake", "batch_size": 1})
    _write_corpus(settings, [f"제{i}조 법률 문장 {i}" for i in range(N)])
    return settings


def _interrupt_second_upload(monkeypatch):
    original = VectorStore.upsert
    calls = 0

    def upsert(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("upload failed")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(VectorStore, "upsert", upsert)
    return original


def _embedded(embedder) -> list[str]:
    return [text for call in embedder.embed_documents.call_args_list for text in call.args[0]]


def _count(settings, embedder) -> int:
    store = open_store(settings, embedder)
    try:
        return store.count()
    finally:
        store.close()


def test_resume_embeds_only_missing_chunks_and_writes_metadata_once_complete(settings, monkeypatch):
    embedder = FakeEmbedder()
    embedder.embed_documents = Mock(wraps=embedder.embed_documents)
    original = _interrupt_second_upload(monkeypatch)
    with pytest.raises(RuntimeError, match="upload failed"):
        build_index(settings, embedder=embedder)
    first = set(_embedded(embedder)[:4])  # the batch that reached the store
    assert not (settings.index_corpus_dir() / "index.json").exists()

    monkeypatch.setattr(VectorStore, "upsert", original)
    embedder.embed_documents.reset_mock()
    report = build_index(settings, embedder=embedder, resume=True)

    assert report.chunks == N and _count(settings, embedder) == N
    resumed = _embedded(embedder)
    assert len(resumed) == N - 4 and not first & set(resumed)
    assert (settings.index_corpus_dir() / "index.json").exists()


def test_resume_of_a_complete_index_embeds_nothing(settings):
    embedder = FakeEmbedder()
    build_index(settings, embedder=embedder)
    embedder.embed_documents = Mock(wraps=embedder.embed_documents)
    assert build_index(settings, embedder=embedder, resume=True).chunks == N
    assert _embedded(embedder) == [] and _count(settings, embedder) == N


def test_resume_refuses_a_changed_corpus(settings, monkeypatch):
    embedder = FakeEmbedder()
    _interrupt_second_upload(monkeypatch)
    with pytest.raises(RuntimeError):
        build_index(settings, embedder=embedder)
    _write_corpus(settings, [f"바뀐 문장 {i}" for i in range(N)])
    with pytest.raises(ValueError, match="cannot resume"):
        build_index(settings, embedder=embedder, resume=True)


def test_resume_refuses_without_saved_statistics_or_collection(settings):
    embedder = FakeEmbedder()
    with pytest.raises(ValueError, match="bm25.json"):
        build_index(settings, embedder=embedder, resume=True)
    build_index(settings, embedder=embedder)
    store = open_store(settings, embedder)
    try:
        store.client.delete_collection(store.collection)
    finally:
        store.close()
    with pytest.raises(ValueError, match="existing collection"):
        build_index(settings, embedder=embedder, resume=True)


def test_resume_refuses_a_different_on_disk_setting(settings):
    embedder = FakeEmbedder()
    build_index(settings, embedder=embedder)
    with pytest.raises(ValueError, match="on-disk"):
        build_index(settings, embedder=embedder, resume=True, on_disk=True)


def _ollama(ps):
    calls = []

    def respond(request):
        calls.append(request.url.path)
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2]] * len(json.loads(request.content)["input"])})
        return ps(request)

    return OllamaEmbedder("bge-m3", "http://test", transport=httpx.MockTransport(respond)), calls


def test_ollama_warns_once_when_documents_embed_on_cpu(caplog):
    embedder, calls = _ollama(lambda _: httpx.Response(200, json={"models": [{"name": "bge-m3:latest", "size_vram": 0}]}))
    embedder.embed_documents(["a"])
    embedder.embed_documents(["b"])
    assert "CPU only" in caplog.text and calls.count("/api/ps") == 1


def test_ollama_placement_check_is_quiet_on_gpu_or_failure_and_skipped_for_queries(caplog):
    gpu, _ = _ollama(lambda _: httpx.Response(200, json={"models": [{"name": "bge-m3:latest", "size_vram": 1200}]}))
    gpu.embed_documents(["a"])
    broken, _ = _ollama(lambda _: httpx.Response(500))
    broken.embed_documents(["a"])
    query, calls = _ollama(lambda _: httpx.Response(200, json={"models": [{"name": "bge-m3", "size_vram": 0}]}))
    query.embed_query("질문")
    assert "CPU only" not in caplog.text and "/api/ps" not in calls

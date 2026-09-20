from datetime import date
from pathlib import Path

from legal_rag.embeddings import FakeEmbedder
from legal_rag.retrieval.dense import DenseRetriever
from legal_rag.retrieval.sparse import BM25Stats
from legal_rag.retrieval.sparse_retriever import SparseRetriever
from legal_rag.retrieval.store import VectorStore
from legal_rag.retrieval.tokenizer import RegexTokenizer
from legal_rag.schemas import Chunk, ChunkMetadata, DocumentType, RetrievalFilters


def _chunks():
    def mk(i, text, doc_type, **meta):
        return Chunk(id=f"d{i}#p", document_id=f"d{i}", document_type=doc_type, path="p", order=0, text=text, title="t",
                     metadata=ChunkMetadata(dataset="f", corpus_version="v", **meta))
    return [
        mk(1, "주택임대차보호법 제3조의2(보증금의 회수)\n임대차 보증금 반환 청구", DocumentType.statute, law_name="주택임대차보호법", article="제3조의2"),
        mk(2, "민법 제750조(불법행위의 내용)\n불법행위 손해배상 책임", DocumentType.statute, law_name="민법", article="제750조"),
        mk(3, "대법원 2018다12345 (2019-05-30) — 판단\n보증금 반환 판결 이유", DocumentType.court_decision, court="대법원", case_number="2018다12345", decision_date=date(2019, 5, 30)),
    ]


def _build(tmp_path: Path):
    chunks = _chunks()
    embedder, tok = FakeEmbedder(32), RegexTokenizer()
    stats = BM25Stats.fit({c.id: tok.tokenize(c.text) for c in chunks}, k1=1.2, b=0.75)
    store = VectorStore.embedded(tmp_path / "q", collection="test", dimension=32)
    store.recreate()
    store.upsert(chunks, dense=embedder.embed_documents([c.text for c in chunks]),
                 sparse=[stats.document_vector(tok.tokenize(c.text)) for c in chunks])
    return store, embedder, tok, stats


def test_dense_retrieval_ranks_similar_text_first(tmp_path):
    store, embedder, *_ = _build(tmp_path)
    hits = DenseRetriever(store, embedder).retrieve("보증금 반환", k=3)
    assert hits[0].chunk.document_id in {"d1", "d3"} and hits[0].retriever == "dense"
    assert store.count() == 3


def test_sparse_retrieval_matches_exact_article(tmp_path):
    store, _, tok, stats = _build(tmp_path)
    hits = SparseRetriever(store, tok, stats).retrieve("민법 제750조", k=3)
    assert hits[0].chunk.metadata.article == "제750조" and hits[0].retriever == "sparse"
    assert SparseRetriever(store, tok, stats).retrieve("zzzz", k=3) == []


def test_filters_restrict_document_type_and_date(tmp_path):
    store, embedder, *_ = _build(tmp_path)
    r = DenseRetriever(store, embedder)
    only_cases = r.retrieve("보증금", k=3, filters=RetrievalFilters(document_types=[DocumentType.court_decision]))
    assert {h.chunk.document_type for h in only_cases} == {DocumentType.court_decision}
    assert r.retrieve("보증금", k=3, filters=RetrievalFilters(decided_after=date(2020, 1, 1))) == []
    assert len(r.retrieve("보증금", k=3, filters=RetrievalFilters(decided_after=date(2019, 1, 1)))) == 1
    assert [h.chunk.id for h in r.retrieve("보증금", k=3, filters=RetrievalFilters(law_name="민법"))] == ["d2#p"]


def test_store_round_trips_chunk_payload(tmp_path):
    store, embedder, *_ = _build(tmp_path)
    hit = DenseRetriever(store, embedder).retrieve("손해배상", k=1)[0]
    assert hit.chunk.metadata.law_name == "민법" and hit.chunk.id == "d2#p"
    assert hit.chunk == _chunks()[1]


def test_server_store_gets_a_write_timeout_long_enough_for_a_large_upsert():
    """qdrant-client defaults to 5 seconds; a 700k-chunk build times out mid-upsert."""
    store = VectorStore.server("http://localhost:6333", None, "c", 4, timeout_seconds=90.0)
    assert store.client._client._timeout == 90.0


def test_open_store_passes_the_configured_timeout():
    from legal_rag.config import Settings
    from legal_rag.factory import open_store

    settings = Settings.model_validate(
        {"qdrant": {"mode": "server", "url": "http://localhost:6333", "timeout_seconds": 77.0}}
    )
    assert open_store(settings, FakeEmbedder(dimension=8)).client._client._timeout == 77.0


class _RecordingClient:
    """Captures create_collection kwargs; the embedded backend does not echo hnsw_config."""

    def __init__(self):
        self.kwargs = {}

    def collection_exists(self, name):
        return False

    def create_collection(self, name, **kwargs):
        self.kwargs = kwargs


def test_recreate_can_keep_vectors_and_the_graph_on_disk():
    """716k 1024-dim vectors do not fit in this host's Qdrant container alongside the
    existing corpora; memory-mapped storage is what makes the build possible at all."""
    store = VectorStore(_RecordingClient(), "c", 8)
    store.recreate(on_disk=True)
    assert store.client.kwargs["vectors_config"][VectorStore.DENSE].on_disk is True
    assert store.client.kwargs["hnsw_config"].on_disk is True


def test_recreate_keeps_vectors_in_memory_by_default(tmp_path):
    store = VectorStore.embedded(tmp_path / "q2", collection="c", dimension=8)
    store.recreate()
    config = store.client.get_collection("c").config
    assert not config.params.vectors[VectorStore.DENSE].on_disk

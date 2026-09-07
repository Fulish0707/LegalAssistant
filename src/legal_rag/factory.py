"""Composition root: build concrete backends from ``Settings``.

Scripts and the API both go through here so that swapping an embedder, store,
reranker, or model is a configuration change, never a code change.
"""

from __future__ import annotations

import json
from pathlib import Path

from legal_rag.config import Settings
from legal_rag.embeddings import Embedder, FakeEmbedder, OllamaEmbedder, slug
from legal_rag.reranking import NoopReranker, Reranker
from legal_rag.retrieval import (
    BM25Stats,
    DenseRetriever,
    HybridRetriever,
    RRFFuser,
    SparseRetriever,
    Tokenizer,
    VectorStore,
    build_tokenizer,
)


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding.backend == "fake":
        return FakeEmbedder(dimension=settings.embedding.fake_dimension)
    if settings.embedding.backend == "ollama":
        return OllamaEmbedder(
            model=settings.embedding.model,
            base_url=settings.embedding.base_url,
            batch_size=settings.embedding.batch_size,
            timeout=settings.embedding.timeout_seconds,
        )
    raise ValueError(f"unknown embedding backend {settings.embedding.backend!r}")


def build_tokenizer_from_settings(settings: Settings) -> Tokenizer:
    return build_tokenizer(settings.sparse.tokenizer)


def collection_name(settings: Settings, embedder: Embedder) -> str:
    return f"legal_{slug(settings.corpus.version)}_{slug(embedder.name)}"


def open_store(settings: Settings, embedder: Embedder) -> VectorStore:
    name = collection_name(settings, embedder)
    if settings.qdrant.mode == "server":
        api_key = settings.qdrant.api_key.get_secret_value() if settings.qdrant.api_key else None
        return VectorStore.server(settings.qdrant.url, api_key, name, embedder.dimension)
    return VectorStore.embedded(settings.qdrant_path(), name, embedder.dimension)


def build_reranker(settings: Settings) -> Reranker:
    if settings.rerank.backend == "noop":
        return NoopReranker()
    if settings.rerank.backend == "cross_encoder":
        from legal_rag.reranking.cross_encoder import CrossEncoderReranker  # heavy import, optional extra

        return CrossEncoderReranker(model=settings.rerank.model, device=settings.rerank.device)
    raise ValueError(f"unknown rerank backend {settings.rerank.backend!r}")


def index_metadata_path(settings: Settings) -> Path:
    return settings.index_corpus_dir() / "index.json"


def read_index_metadata(settings: Settings) -> dict:
    path = index_metadata_path(settings)
    if not path.exists():
        raise FileNotFoundError(f"no index at {settings.index_corpus_dir()}; run scripts/build_index.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def build_hybrid_retriever(settings: Settings, *, embedder: Embedder | None = None, store: VectorStore | None = None) -> HybridRetriever:
    embedder = embedder or build_embedder(settings)
    store = store or open_store(settings, embedder)
    tokenizer = build_tokenizer_from_settings(settings)
    stats = BM25Stats.load(settings.index_corpus_dir() / "bm25.json")
    legs = [DenseRetriever(store, embedder), SparseRetriever(store, tokenizer, stats)]
    fuser = RRFFuser(
        rrf_k=settings.retrieval.rrf_k,
        weights={"dense": settings.retrieval.dense_weight, "sparse": settings.retrieval.sparse_weight},
    )
    return HybridRetriever(
        legs,
        fuser,
        build_reranker(settings),
        k_leg=max(settings.retrieval.k_dense, settings.retrieval.k_sparse),
        k_fused=settings.retrieval.k_fused,
    )


def build_chat_model(settings: Settings):
    from legal_rag.llm import FakeChatModel, OllamaChatModel

    if settings.llm.backend == "fake":
        return FakeChatModel({})
    if settings.llm.backend == "ollama":
        return OllamaChatModel(model=settings.llm.model, base_url=settings.llm.base_url, timeout=settings.llm.timeout_seconds)
    if settings.llm.backend == "transformers":
        raise RuntimeError("the transformers backend arrives with V0.4 (QLoRA adapters)")
    raise ValueError(f"unknown llm backend {settings.llm.backend!r}")


def build_orchestrator(settings: Settings, *, retriever: HybridRetriever | None = None):
    from legal_rag.rag.conversation import SQLiteConversationStore
    from legal_rag.rag.orchestrator import Orchestrator
    from legal_rag.verification import VerificationPolicy, Verifier

    settings.index_corpus_dir().mkdir(parents=True, exist_ok=True)
    return Orchestrator.from_components(
        settings=settings,
        retriever=retriever or build_hybrid_retriever(settings),
        model=build_chat_model(settings),
        store=SQLiteConversationStore(settings.index_corpus_dir() / "conversations.db"),
        verifier=Verifier(),
        policy=VerificationPolicy(max_regenerations=settings.rag.max_regenerations),
    )

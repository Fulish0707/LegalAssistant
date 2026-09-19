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
    Retriever,
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


class IndexCompatibilityError(RuntimeError):
    """Current settings cannot query the index on disk."""


def check_index_compatibility(settings: Settings, *, embedder_name: str | None = None) -> dict:
    """Refuse to query an index that was built under incompatible settings.

    BM25 splits its state: stored document vectors were encoded with the indexing
    tokenizer, while queries are tokenized now. Analyze a query differently and its
    terms miss the document-frequency map entirely, so ``query_vector`` drops them and
    the sparse leg returns nothing — while dense retrieval keeps returning plausible
    hits and hides the failure. Fail closed instead.
    """
    metadata = read_index_metadata(settings)
    index_dir = settings.index_corpus_dir()
    problems: list[str] = []
    if not (index_dir / "bm25.json").exists():
        problems.append(f"bm25.json is missing from {index_dir}")
    recorded_tokenizer = metadata.get("tokenizer")
    if recorded_tokenizer != settings.sparse.tokenizer:
        problems.append(
            f"tokenizer: index built with {recorded_tokenizer!r}, settings say {settings.sparse.tokenizer!r}"
        )
    if embedder_name is not None and metadata.get("embedder") != embedder_name:
        problems.append(f"embedder: index built with {metadata.get('embedder')!r}, settings say {embedder_name!r}")
    if problems:
        raise IndexCompatibilityError(
            f"index at {index_dir} does not match the current settings:\n  "
            + "\n  ".join(problems)
            + "\nRestore the recorded settings, or rebuild with scripts/build_index.py."
        )
    return metadata


def build_hybrid_retriever(
    settings: Settings, *, embedder: Embedder | None = None, reranker: Reranker | None = None
) -> HybridRetriever:
    """Supply an ``embedder`` or ``reranker`` to share one instance across several retrievers;
    a cross-encoder is a GPU-resident model and one copy per corpus would not fit."""
    embedder = embedder or build_embedder(settings)
    legs = build_corpus_legs(settings, embedder=embedder)
    return HybridRetriever(
        legs,
        RRFFuser(rrf_k=settings.retrieval.rrf_k, weights=fusion_weights(settings)),
        reranker or build_reranker(settings),
        k_leg=max(settings.retrieval.k_dense, settings.retrieval.k_sparse),
        k_fused=settings.retrieval.k_fused,
    )


def corpus_settings(settings: Settings, corpus_version: str) -> Settings:
    return settings.model_copy(
        update={"corpus": settings.corpus.model_copy(update={"version": corpus_version})}
    )


def build_corpus_legs(settings: Settings, *, embedder: Embedder | None = None) -> list[Retriever]:
    """Dense and sparse legs for every configured corpus, each named after its corpus.

    Each corpus keeps its own collection and its own BM25 statistics, so a term missing
    from one corpus's document frequencies simply yields nothing from that corpus rather
    than being dropped from the query everywhere.
    """
    embedder = embedder or build_embedder(settings)
    versions = settings.retrieval.corpora or [settings.corpus.version]
    legs: list[Retriever] = []
    for version in versions:
        scoped = corpus_settings(settings, version)
        check_index_compatibility(scoped, embedder_name=embedder.name)
        store = open_store(scoped, embedder)
        stats = BM25Stats.load(scoped.index_corpus_dir() / "bm25.json")
        dense = DenseRetriever(store, embedder)
        sparse = SparseRetriever(store, build_tokenizer_from_settings(scoped), stats)
        dense.name = f"{version}:dense"
        sparse.name = f"{version}:sparse"
        legs += [dense, sparse]
    return legs


def select_legs(retriever: HybridRetriever, leg: str) -> HybridRetriever:
    """Narrow a retriever to one kind of leg across every corpus, for diagnostics."""
    if leg == "hybrid":
        return retriever
    chosen = [candidate for candidate in retriever.legs if candidate.name.rsplit(":", 1)[-1] == leg]
    if not chosen:
        raise ValueError(f"no {leg!r} leg; available: {[c.name for c in retriever.legs]}")
    narrowed = HybridRetriever(chosen, retriever.fuser, retriever.reranker,
                               k_leg=retriever.k_leg, k_fused=retriever.k_fused)
    narrowed.name = leg  # so an evaluation report records which leg it measured
    return narrowed


def close_legs(retriever: HybridRetriever) -> None:
    """Close each distinct store once; a corpus's two legs share one store."""
    for store in {id(leg.store): leg.store for leg in retriever.legs if hasattr(leg, "store")}.values():
        store.close()


def fusion_weights(settings: Settings) -> dict[str, float]:
    """Per-leg RRF weights: the corpus weight scales both of that corpus's legs."""
    versions = settings.retrieval.corpora or [settings.corpus.version]
    weights: dict[str, float] = {}
    for version in versions:
        corpus_weight = settings.retrieval.corpus_weights.get(version, 1.0)
        weights[f"{version}:dense"] = corpus_weight * settings.retrieval.dense_weight
        weights[f"{version}:sparse"] = corpus_weight * settings.retrieval.sparse_weight
    return weights


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

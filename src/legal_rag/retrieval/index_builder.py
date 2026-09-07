"""Processed corpus → BM25 statistics + dense vectors → Qdrant collection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, TypeVar

from legal_rag.config import Settings
from legal_rag.embeddings import Embedder
from legal_rag.factory import build_embedder, build_tokenizer_from_settings, collection_name, index_metadata_path, open_store
from legal_rag.ingestion.corpus import Corpus
from legal_rag.schemas import Chunk

from .sparse import BM25Stats

T = TypeVar("T")


@dataclass(frozen=True)
class IndexReport:
    chunks: int
    collection: str
    dimension: int
    index_dir: Path


def batched(items: list[T], size: int) -> Iterable[list[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_index(settings: Settings, *, embedder: Embedder | None = None, progress: Callable[[int, int], None] | None = None) -> IndexReport:
    corpus = Corpus.load(settings.processed_corpus_dir())
    chunks: list[Chunk] = list(corpus.chunks())
    embedder = embedder or build_embedder(settings)
    tokenizer = build_tokenizer_from_settings(settings)

    tokenized = {chunk.id: tokenizer.tokenize(chunk.text) for chunk in chunks}
    stats = BM25Stats.fit(tokenized, k1=settings.sparse.k1, b=settings.sparse.b)
    index_dir = settings.index_corpus_dir()
    index_dir.mkdir(parents=True, exist_ok=True)
    stats.save(index_dir / "bm25.json")

    store = open_store(settings, embedder)
    store.recreate()
    done = 0
    for batch in batched(chunks, max(1, settings.embedding.batch_size * 4)):
        dense = embedder.embed_documents([chunk.text for chunk in batch])
        sparse = [stats.document_vector(tokenized[chunk.id]) for chunk in batch]
        store.upsert(batch, dense=dense, sparse=sparse)
        done += len(batch)
        if progress:
            progress(done, len(chunks))
    store.close()

    metadata = {
        "collection": collection_name(settings, embedder),
        "embedder": embedder.name,
        "dimension": embedder.dimension,
        "tokenizer": tokenizer.name,
        "chunks": len(chunks),
        "corpus_version": settings.corpus.version,
        "config_hash": settings.hash(),
        "qdrant_mode": settings.qdrant.mode,
    }
    index_metadata_path(settings).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return IndexReport(len(chunks), metadata["collection"], embedder.dimension, index_dir)

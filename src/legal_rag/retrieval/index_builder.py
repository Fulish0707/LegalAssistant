"""Processed corpus → BM25 statistics + dense vectors → Qdrant collection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Callable, Iterable, TypeVar

from legal_rag.config import Settings
from legal_rag.embeddings import Embedder
from legal_rag.factory import build_embedder, build_tokenizer_from_settings, collection_name, index_metadata_path, open_store
from legal_rag.ingestion.corpus import Corpus

from .sparse import BM25Stats

T = TypeVar("T")


@dataclass(frozen=True)
class IndexReport:
    chunks: int
    collection: str
    dimension: int
    index_dir: Path


def batched(items: Iterable[T], size: int) -> Iterable[list[T]]:
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


def build_index(settings: Settings, *, embedder: Embedder | None = None, on_disk: bool = False,
                progress: Callable[[int, int], None] | None = None) -> IndexReport:
    corpus = Corpus.load(settings.processed_corpus_dir())
    embedder = embedder or build_embedder(settings)
    tokenizer = build_tokenizer_from_settings(settings)

    stats = BM25Stats.fit_stream(
        (tokenizer.tokenize(chunk.text) for chunk in corpus.chunks()),
        k1=settings.sparse.k1, b=settings.sparse.b,
    )
    index_dir = settings.index_corpus_dir()
    index_dir.mkdir(parents=True, exist_ok=True)
    stats.save(index_dir / "bm25.json")

    store = open_store(settings, embedder)
    done = 0
    try:
        store.recreate(on_disk=on_disk)
        for batch in batched(corpus.chunks(), max(1, settings.embedding.batch_size * 4)):
            dense = embedder.embed_documents([chunk.text for chunk in batch])
            sparse = [stats.document_vector(tokenizer.tokenize(chunk.text)) for chunk in batch]
            store.upsert(batch, dense=dense, sparse=sparse)
            done += len(batch)
            if progress:
                progress(done, stats.n_docs)
    finally:
        store.close()

    metadata = {
        "collection": collection_name(settings, embedder),
        "embedder": embedder.name,
        "dimension": embedder.dimension,
        "tokenizer": tokenizer.name,
        "chunks": done,
        "corpus_version": settings.corpus.version,
        "config_hash": settings.hash(),
        "qdrant_mode": settings.qdrant.mode,
        "on_disk": on_disk,
    }
    index_metadata_path(settings).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return IndexReport(done, metadata["collection"], embedder.dimension, index_dir)

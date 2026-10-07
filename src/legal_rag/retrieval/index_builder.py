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


def build_index(settings: Settings, *, embedder: Embedder | None = None, on_disk: bool = False, resume: bool = False,
                progress: Callable[[int, int], None] | None = None) -> IndexReport:
    corpus = Corpus.load(settings.processed_corpus_dir())
    embedder = embedder or build_embedder(settings)
    tokenizer = build_tokenizer_from_settings(settings)

    index_dir = settings.index_corpus_dir()
    stats_path = index_dir / "bm25.json"
    saved_stats = BM25Stats.load(stats_path) if resume and stats_path.exists() else None
    if resume and saved_stats is None:
        raise ValueError("Resume requires the original bm25.json; it must not be regenerated under new settings.")
    stats = BM25Stats.fit_stream(
        (tokenizer.tokenize(chunk.text) for chunk in corpus.chunks()),
        k1=settings.sparse.k1, b=settings.sparse.b,
    )
    if resume and vars(stats) != vars(saved_stats):
        raise ValueError("Corpus/tokenizer/BM25 statistics changed; cannot resume this index.")
    del saved_stats
    index_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        stats.save(stats_path)

    store = open_store(settings, embedder)
    done = 0
    try:
        if resume:
            if not store.exists():
                raise ValueError("Resume requires an existing collection; no collection will be created or cleared.")
            info = store.client.get_collection(store.collection)
            vectors = info.config.params.vectors
            if (not isinstance(vectors, dict) or store.DENSE not in vectors
                    or vectors[store.DENSE].size != embedder.dimension
                    or vectors[store.DENSE].distance != "Cosine"
                    or store.SPARSE not in (info.config.params.sparse_vectors or {})):
                raise ValueError("Existing vector configuration is incompatible; cannot resume.")
            if bool(vectors[store.DENSE].on_disk) != on_disk:
                raise ValueError("Resume must use the original --on-disk setting.")
            if store.count() > stats.n_docs:
                raise ValueError("Collection has more points than this corpus; cannot resume.")
        else:
            store.recreate(on_disk=on_disk)
        batch_size = max(256 if resume else 1, settings.embedding.batch_size * 4)
        for batch in batched(corpus.chunks(), batch_size):
            matched = store.matching_chunk_ids(batch) if resume else set()
            missing = [chunk for chunk in batch if chunk.id not in matched]
            if missing:
                dense = embedder.embed_documents([chunk.text for chunk in missing])
                sparse = [stats.document_vector(tokenizer.tokenize(chunk.text)) for chunk in missing]
                store.upsert(missing, dense=dense, sparse=sparse)
            done += len(batch)
            if progress:
                progress(done, stats.n_docs)
        if resume and store.count() != done:
            raise ValueError("Final collection count does not match the corpus; completion metadata was not written.")
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

"""Shared chunk construction. Chunk text always begins with a structural header line."""

from __future__ import annotations

from typing import Any

from legal_rag.schemas import Chunk, ChunkMetadata, LegalDocument

STATUTE_MAX_CHARS = 800
DECISION_MAX_CHARS = 1200
INTERPRETATION_MAX_CHARS = 1200


def make_chunk(doc: LegalDocument, *, path: str, order: int, text: str, corpus_version: str, **meta: Any) -> Chunk:
    metadata = ChunkMetadata(
        dataset=doc.source.dataset,
        corpus_version=corpus_version,
        legal_domain=doc.legal_domain,
        effective_date=doc.effective_date,
        **meta,
    )
    return Chunk(
        id=f"{doc.id}#{path}",
        document_id=doc.id,
        document_type=doc.document_type,
        path=path,
        order=order,
        text=text,
        title=doc.title,
        metadata=metadata,
    )

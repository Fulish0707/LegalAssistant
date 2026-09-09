"""Pick the legal-aware chunker for a document type."""

from __future__ import annotations

from legal_rag.schemas import AdminInterpretation, Chunk, CourtDecision, LegalDocument, Statute

from .decision import chunk_decision
from .fallback import chunk_fallback
from .interpretation import chunk_interpretation
from .statute import chunk_statute


def chunk_document(doc: LegalDocument, *, corpus_version: str) -> list[Chunk]:
    if isinstance(doc, Statute):
        return chunk_statute(doc, corpus_version)
    if isinstance(doc, CourtDecision):
        return chunk_decision(doc, corpus_version)
    if isinstance(doc, AdminInterpretation):
        return chunk_interpretation(doc, corpus_version)
    return chunk_fallback(doc, corpus_version)

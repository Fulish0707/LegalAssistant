"""Administrative interpretation chunking: question and answer together when they fit."""

from __future__ import annotations

from legal_rag.schemas import AdminInterpretation, Chunk

from .base import INTERPRETATION_MAX_CHARS, make_chunk
from .fallback import chunk_text_windows


def chunk_interpretation(doc: AdminInterpretation, corpus_version: str, max_chars: int = INTERPRETATION_MAX_CHARS) -> list[Chunk]:
    date_label = doc.date.isoformat() if doc.date else "날짜 미상"
    header = f"{doc.authority} {doc.reference_number} ({date_label}) — {doc.subject}"
    meta = {"authority": doc.authority, "reference_number": doc.reference_number}
    combined = f"[질의] {doc.question}\n[회답] {doc.interpretation}"
    if len(header) + 1 + len(combined) <= max_chars:
        return [make_chunk(doc, path="sec:qa", order=0, text=f"{header}\n{combined}", corpus_version=corpus_version, section="qa", **meta)]
    chunks = chunk_text_windows(doc, doc.question, header=f"{header} — 질의", path_prefix="sec:question", start_order=0, corpus_version=corpus_version, max_chars=max_chars, section="question", **meta)
    chunks += chunk_text_windows(doc, doc.interpretation, header=f"{header} — 회답", path_prefix="sec:interpretation", start_order=len(chunks), corpus_version=corpus_version, max_chars=max_chars, section="interpretation", **meta)
    return chunks

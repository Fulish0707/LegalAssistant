"""Sentence-window chunking used when a legal unit is too long or has no structure."""

from __future__ import annotations

from typing import Any

from legal_rag.schemas import Chunk, LegalDocument
from legal_rag.textutil.sentences import split_sentences

from .base import make_chunk


def window_sentences(sentences: list[str], max_chars: int) -> list[str]:
    """Greedy windows of whole sentences, overlapping by one sentence."""
    windows: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if current and len(" ".join(current + [sentence])) > max_chars:
            windows.append(" ".join(current))
            current = [current[-1], sentence]
        else:
            current.append(sentence)
    if current:
        windows.append(" ".join(current))
    return windows


def chunk_text_windows(
    doc: LegalDocument,
    text: str,
    *,
    header: str,
    path_prefix: str,
    start_order: int,
    corpus_version: str,
    max_chars: int,
    **meta: Any,
) -> list[Chunk]:
    """Emit ``path_prefix`` when the text fits, otherwise ``path_prefix/1``, ``/2``, ..."""
    body = text.strip()
    if len(header) + 1 + len(body) <= max_chars:
        return [make_chunk(doc, path=path_prefix, order=start_order, text=f"{header}\n{body}", corpus_version=corpus_version, **meta)]
    windows = window_sentences(split_sentences(body), max_chars=max(max_chars - len(header) - 1, 1))
    return [
        make_chunk(doc, path=f"{path_prefix}/{n}", order=start_order + n - 1, text=f"{header}\n{window}", corpus_version=corpus_version, **meta)
        for n, window in enumerate(windows, start=1)
    ]


def chunk_fallback(doc: LegalDocument, corpus_version: str, max_chars: int = 1000) -> list[Chunk]:
    return chunk_text_windows(doc, doc.text, header=doc.title, path_prefix="fallback", start_order=0, corpus_version=corpus_version, max_chars=max_chars)

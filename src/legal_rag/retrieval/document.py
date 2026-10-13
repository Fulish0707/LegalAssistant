"""Rebuild the whole text of a document from its stored chunks."""

from __future__ import annotations

from legal_rag.schemas import Chunk

from .store import VectorStore

# Windowed corpora repeat about 400 characters between neighbours; a shorter match may be chance.
MIN_OVERLAP, MAX_OVERLAP = 10, 1000


def full_text(chunks: list[Chunk], cited_chunk_id: str) -> tuple[str, int, int]:
    """(text, start, end): the chunks joined in order, and where the cited chunk lies in the text.

    A chunk cut as an overlapping window starts with the end of the chunk before it; the repeated
    part is written once. Chunks that do not overlap, such as statute articles, are set apart.
    """
    text, cited = "", (0, 0)
    for chunk in chunks:
        part = chunk.text
        overlap = next((size for size in range(min(len(text), len(part), MAX_OVERLAP), MIN_OVERLAP - 1, -1)
                        if text.endswith(part[:size])), 0)
        if text and not overlap:
            text += "\n\n"
        start = len(text) - overlap
        text += part[overlap:]
        if chunk.id == cited_chunk_id:
            cited = (start, len(text))
    return text, *cited


def find_document(stores: list[VectorStore], chunk_id: str) -> tuple[Chunk, str, int, int] | None:
    """The cited chunk and its document's full text, from the first corpus that holds the chunk."""
    for store in stores:
        chunk = store.chunk(chunk_id)
        if chunk is not None:
            return chunk, *full_text(store.document_chunks(chunk), chunk_id)
    return None

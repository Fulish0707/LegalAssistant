"""Embedder protocol. Implementations must not know about Qdrant."""

from __future__ import annotations

import re
from typing import Protocol


class Embedder(Protocol):
    name: str  # e.g. "ollama:bge-m3"; stamped into collection names
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

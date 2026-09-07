"""Deterministic hash embeddings for offline tests. Similar texts share tokens, so they score higher."""

from __future__ import annotations

import hashlib
import math

from legal_rag.retrieval.tokenizer import RegexTokenizer


class FakeEmbedder:
    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension
        self.name = f"fake:{dimension}"
        self._tokenizer = RegexTokenizer()

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in self._tokenizer.tokenize(text):
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            slot = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[slot] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

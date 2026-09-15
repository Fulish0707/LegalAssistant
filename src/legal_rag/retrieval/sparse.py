"""BM25 statistics expressed as sparse vectors.

Document vectors carry the BM25 term weight (tf saturation and length norm);
query vectors carry idf. Qdrant's sparse dot product is then exactly BM25.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


def term_id(term: str) -> int:
    return int.from_bytes(hashlib.sha1(term.encode("utf-8")).digest()[:4], "big")


@dataclass(frozen=True)
class SparseVec:
    indices: list[int]
    values: list[float]


class BM25Stats:
    def __init__(self, df: dict[str, int], n_docs: int, avg_len: float, k1: float, b: float) -> None:
        self.df = df
        self.n_docs = n_docs
        self.avg_len = avg_len
        self.k1 = k1
        self.b = b

    @classmethod
    def fit(cls, tokenized: dict[str, list[str]], *, k1: float, b: float) -> "BM25Stats":
        return cls.fit_stream(tokenized.values(), k1=k1, b=b)

    @classmethod
    def fit_stream(cls, documents: Iterable[list[str]], *, k1: float, b: float) -> "BM25Stats":
        """Fit corpus frequencies without retaining each document token list."""
        df: Counter[str] = Counter()
        total = 0
        n_docs = 0
        for tokens in documents:
            n_docs += 1
            total += len(tokens)
            df.update(set(tokens))
        return cls(dict(df), n_docs, total / n_docs if n_docs else 0.0, k1, b)

    def idf(self, term: str) -> float:
        d = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - d + 0.5) / (d + 0.5))

    def document_vector(self, tokens: list[str]) -> SparseVec:
        tf = Counter(tokens)
        length = len(tokens)
        avg = self.avg_len or 1.0
        pairs: dict[int, float] = {}
        for term, freq in tf.items():
            weight = freq * (self.k1 + 1) / (freq + self.k1 * (1 - self.b + self.b * length / avg))
            key = term_id(term)
            pairs[key] = pairs.get(key, 0.0) + weight
        indices = sorted(pairs)
        return SparseVec(indices, [pairs[i] for i in indices])

    def query_vector(self, tokens: list[str]) -> SparseVec:
        pairs: dict[int, float] = {}
        for term in set(tokens):
            if term in self.df:
                pairs[term_id(term)] = self.idf(term)
        indices = sorted(pairs)
        return SparseVec(indices, [pairs[i] for i in indices])

    def save(self, path: Path) -> None:
        payload = {"df": self.df, "n_docs": self.n_docs, "avg_len": self.avg_len, "k1": self.k1, "b": self.b}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25Stats":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["df"], data["n_docs"], data["avg_len"], data["k1"], data["b"])

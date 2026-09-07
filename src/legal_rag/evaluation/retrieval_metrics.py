"""Pure ranking metrics over a ranked id list and a relevant id set."""

from __future__ import annotations

import math
from typing import Iterable


def recall_at_k(ranked: list[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    return len(set(ranked[:k]) & rel) / len(rel)


def precision_at_k(ranked: list[str], relevant: Iterable[str], k: int) -> float:
    if k <= 0:
        return 0.0
    rel = set(relevant)
    return len(set(ranked[:k]) & rel) / k


def mrr(ranked: list[str], relevant: Iterable[str]) -> float:
    rel = set(relevant)
    for position, item in enumerate(ranked, start=1):
        if item in rel:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    dcg = sum(1.0 / math.log2(position + 1) for position, item in enumerate(ranked[:k], start=1) if item in rel)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0

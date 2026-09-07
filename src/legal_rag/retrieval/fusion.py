"""Reciprocal rank fusion over any number of ranked lists."""

from __future__ import annotations

from typing import Protocol

from legal_rag.schemas import Chunk, ScoredChunk


class Fuser(Protocol):
    def fuse(self, ranked_lists: list[list[ScoredChunk]], k: int) -> list[ScoredChunk]: ...


class RRFFuser:
    def __init__(self, rrf_k: int = 60, weights: dict[str, float] | None = None) -> None:
        self.rrf_k = rrf_k
        self.weights = weights or {}

    def fuse(self, ranked_lists: list[list[ScoredChunk]], k: int) -> list[ScoredChunk]:
        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for ranked in ranked_lists:
            for rank, hit in enumerate(ranked, start=1):
                weight = self.weights.get(hit.retriever, 1.0)
                scores[hit.chunk.id] = scores.get(hit.chunk.id, 0.0) + weight / (self.rrf_k + rank)
                chunks.setdefault(hit.chunk.id, hit.chunk)
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [ScoredChunk(chunk=chunks[chunk_id], score=score, retriever="fused") for chunk_id, score in ordered if score > 0]

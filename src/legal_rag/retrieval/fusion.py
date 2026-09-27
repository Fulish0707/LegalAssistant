"""Reciprocal rank fusion over any number of ranked lists."""

from __future__ import annotations

from typing import Protocol

from legal_rag.schemas import Chunk, ScoredChunk


class Fuser(Protocol):
    def fuse(self, ranked_lists: list[list[ScoredChunk]], k: int) -> list[ScoredChunk]: ...


class RRFFuser:
    """Weighted RRF; ``leg_floor`` reserves each list's top hits a place in the output.

    Under multi-corpus fan-out a chunk can appear only in its own corpus's two legs, so its
    fused score mostly measures whether that corpus's dense and sparse legs agree. Any chunk
    both legs return outscores the rank-1 hit of a single list, and a corpus whose legs disagree
    can be shut out of the candidates the reranker sees (docs/retrieval-diagnostics.md,
    "Why conversational queries missed civil judgments"). With ``leg_floor=n`` the top ``n``
    hits of every positively weighted list are admitted first, rank by rank across lists, and
    the remaining slots go by fused score. The floor fills at most half of ``k``: a multi-query
    chat search fuses queries x legs lists, and 3 from each of 32 lists would otherwise turn
    fusion into round-robin. The output is still ordered by fused score, and ``leg_floor=0``
    is plain RRF.
    """

    def __init__(self, rrf_k: int = 60, weights: dict[str, float] | None = None, leg_floor: int = 0) -> None:
        self.rrf_k = rrf_k
        self.weights = weights or {}
        self.leg_floor = leg_floor

    def fuse(self, ranked_lists: list[list[ScoredChunk]], k: int) -> list[ScoredChunk]:
        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for ranked in ranked_lists:
            for rank, hit in enumerate(ranked, start=1):
                weight = self.weights.get(hit.retriever, 1.0)
                scores[hit.chunk.id] = scores.get(hit.chunk.id, 0.0) + weight / (self.rrf_k + rank)
                chunks.setdefault(hit.chunk.id, hit.chunk)
        ordered = [chunk_id for chunk_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0])) if score > 0]
        chosen = self._floor(ranked_lists, k, scores)
        for chunk_id in ordered:
            if len(chosen) >= k:
                break
            chosen.setdefault(chunk_id, None)
        selected = sorted(chosen, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
        return [ScoredChunk(chunk=chunks[chunk_id], score=scores[chunk_id], retriever="fused") for chunk_id in selected]

    def _floor(self, ranked_lists: list[list[ScoredChunk]], k: int, scores: dict[str, float]) -> dict[str, None]:
        chosen: dict[str, None] = {}  # insertion-ordered set
        cap = k // 2
        for rank in range(self.leg_floor):
            for ranked in ranked_lists:
                if len(chosen) >= cap:
                    return chosen
                if rank < len(ranked) and scores[ranked[rank].chunk.id] > 0 \
                        and self.weights.get(ranked[rank].retriever, 1.0) > 0:
                    chosen.setdefault(ranked[rank].chunk.id, None)
        return chosen

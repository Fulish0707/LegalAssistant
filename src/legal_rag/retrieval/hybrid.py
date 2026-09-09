"""Run every search query through every retrieval leg, fuse, then rerank."""

from __future__ import annotations

from typing import Protocol

from legal_rag.reranking import Reranker
from legal_rag.schemas import RetrievalFilters, ScoredChunk

from .fusion import Fuser


class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, k: int, filters: RetrievalFilters | None = None) -> list[ScoredChunk]: ...


class HybridRetriever:
    name = "hybrid"

    def __init__(self, legs: list[Retriever], fuser: Fuser, reranker: Reranker, *, k_leg: int, k_fused: int) -> None:
        self.legs = legs
        self.fuser = fuser
        self.reranker = reranker
        self.k_leg = k_leg
        self.k_fused = k_fused
        self.last_trace: dict[str, list[tuple[str, float]]] = {}

    def retrieve(self, query: str, k: int, filters: RetrievalFilters | None = None) -> list[ScoredChunk]:
        return self.retrieve_many([query], k, filters)

    def retrieve_many(self, queries: list[str], k: int, filters: RetrievalFilters | None = None, rerank_query: str | None = None) -> list[ScoredChunk]:
        self.last_trace = {}
        ranked_lists: list[list[ScoredChunk]] = []
        for query in queries:
            for leg in self.legs:
                hits = leg.retrieve(query, self.k_leg, filters)
                self.last_trace[f"{leg.name}:{query}"] = [(hit.chunk.id, hit.score) for hit in hits]
                ranked_lists.append(hits)
        fused = self.fuser.fuse(ranked_lists, self.k_fused)
        self.last_trace["fused"] = [(hit.chunk.id, hit.score) for hit in fused]
        reranked = self.reranker.rerank(rerank_query or (queries[0] if queries else ""), fused, k)
        self.last_trace["reranked"] = [(hit.chunk.id, hit.score) for hit in reranked]
        return reranked

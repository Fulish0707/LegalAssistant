"""Run every search query through every retrieval leg, fuse, then rerank."""

from __future__ import annotations

from typing import Protocol

from legal_rag.reranking import Reranker
from legal_rag.schemas import Chunk, RetrievalFilters, ScoredChunk

from .dense import DenseRetriever
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

    def neighbor_chunks(self, chunk: Chunk, radius: int) -> list[Chunk]:
        """Use the hit's corpus store; the dense/sparse legs share that store."""
        version = chunk.metadata.corpus_version
        for leg in self.legs:
            if leg.name.rsplit(":", 1)[0] == version and hasattr(leg, "store"):
                return leg.store.chunks_near(chunk, radius)
        raise ValueError(f"No neighbor store for corpus {version!r}.")

    def retrieve(self, query: str, k: int, filters: RetrievalFilters | None = None) -> list[ScoredChunk]:
        return self.retrieve_many([query], k, filters)

    def retrieve_many(self, queries: list[str], k: int, filters: RetrievalFilters | None = None, rerank_query: str | None = None) -> list[ScoredChunk]:
        self.last_trace = {}
        ranked_lists: list[list[ScoredChunk]] = []
        # All corpora built by the factory share an embedder. Reuse its query vector
        # within this call only; other requests and other embedders remain independent.
        query_vectors: dict[tuple[int, str], list[float]] = {}
        for query in queries:
            for leg in self.legs:
                # A custom subclass may override retrieve(); keep its own behavior.
                if type(leg) is DenseRetriever:
                    key = (id(leg.embedder), query)
                    if key not in query_vectors:
                        query_vectors[key] = leg.embedder.embed_query(query)
                    hits = leg.retrieve_vector(query_vectors[key], self.k_leg, filters)
                else:
                    hits = leg.retrieve(query, self.k_leg, filters)
                self.last_trace[f"{leg.name}:{query}"] = [(hit.chunk.id, hit.score) for hit in hits]
                ranked_lists.append(hits)
        fused = self.fuser.fuse(ranked_lists, self.k_fused)
        self.last_trace["fused"] = [(hit.chunk.id, hit.score) for hit in fused]
        reranked = self.reranker.rerank(rerank_query or (queries[0] if queries else ""), fused, k)
        self.last_trace["reranked"] = [(hit.chunk.id, hit.score) for hit in reranked]
        return reranked

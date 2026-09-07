from legal_rag.schemas import ScoredChunk


class NoopReranker:
    """Keeps fusion order; marks the hits as the final ranked list."""

    name = "noop"

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        return [candidate.model_copy(update={"retriever": "reranked"}) for candidate in candidates[:k]]

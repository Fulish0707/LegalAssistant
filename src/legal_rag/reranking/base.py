from typing import Protocol

from legal_rag.schemas import ScoredChunk


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]: ...

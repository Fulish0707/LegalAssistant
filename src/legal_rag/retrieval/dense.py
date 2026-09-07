from legal_rag.embeddings import Embedder
from legal_rag.schemas import RetrievalFilters, ScoredChunk

from .store import VectorStore


class DenseRetriever:
    name = "dense"

    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, k: int, filters: RetrievalFilters | None = None) -> list[ScoredChunk]:
        vector = self.embedder.embed_query(query)
        return [ScoredChunk(chunk=chunk, score=score, retriever=self.name) for chunk, score in self.store.search_dense(vector, k, filters)]

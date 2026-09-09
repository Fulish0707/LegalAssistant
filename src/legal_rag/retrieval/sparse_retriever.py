from legal_rag.schemas import RetrievalFilters, ScoredChunk

from .sparse import BM25Stats
from .store import VectorStore
from .tokenizer import Tokenizer


class SparseRetriever:
    name = "sparse"

    def __init__(self, store: VectorStore, tokenizer: Tokenizer, stats: BM25Stats) -> None:
        self.store = store
        self.tokenizer = tokenizer
        self.stats = stats

    def retrieve(self, query: str, k: int, filters: RetrievalFilters | None = None) -> list[ScoredChunk]:
        vector = self.stats.query_vector(self.tokenizer.tokenize(query))
        if not vector.indices:
            return []
        hits = self.store.search_sparse(vector, k, filters)
        return [ScoredChunk(chunk=chunk, score=score, retriever=self.name) for chunk, score in hits if score > 0]

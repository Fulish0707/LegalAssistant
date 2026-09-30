"""The only module that talks to Qdrant.

One collection per (corpus version, embedder) holds a named dense vector and a
named sparse BM25 vector per chunk, with the full chunk as payload so hits can
be rebuilt without a second store.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models

from legal_rag.schemas import Chunk, RetrievalFilters

from .filters import MatchAny, Range, date_int, to_conditions
from .sparse import SparseVec

_DERIVED_KEYS = ("decision_date_int", "effective_date_int")


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class VectorStore:
    DENSE = "dense"
    SPARSE = "bm25"

    def __init__(self, client: QdrantClient, collection: str, dimension: int, *, server: bool = False) -> None:
        self.client = client
        self.collection = collection
        self.dimension = dimension
        self._server = server

    @classmethod
    def embedded(cls, path: Path, collection: str, dimension: int) -> "VectorStore":
        Path(path).mkdir(parents=True, exist_ok=True)
        return cls(QdrantClient(path=str(path)), collection, dimension)

    @classmethod
    def server(cls, url: str, api_key: str | None, collection: str, dimension: int,
               *, timeout_seconds: float = 120.0) -> "VectorStore":
        client = QdrantClient(url=url, api_key=api_key, timeout=timeout_seconds)
        return cls(client, collection, dimension, server=True)

    # Lifecycle ------------------------------------------------------------------
    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def recreate(self, *, on_disk: bool = False) -> None:
        """``on_disk`` memory-maps the vectors and the HNSW graph instead of holding them
        in RAM: slower to query, but the only way a corpus larger than the server's memory
        can be indexed at all."""
        if self.exists():
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config={self.DENSE: models.VectorParams(size=self.dimension, distance=models.Distance.COSINE,
                                                            on_disk=on_disk or None)},
            sparse_vectors_config={self.SPARSE: models.SparseVectorParams()},
            hnsw_config=models.HnswConfigDiff(on_disk=True) if on_disk else None,
        )
        if self._server:  # payload indexes are a no-op in embedded mode
            for key in ("document_type", "metadata.law_name", "metadata.court", "metadata.legal_domain"):
                self.client.create_payload_index(self.collection, key, models.PayloadSchemaType.KEYWORD)
            for key in _DERIVED_KEYS:
                self.client.create_payload_index(self.collection, key, models.PayloadSchemaType.INTEGER)

    def count(self) -> int:
        return self.client.count(self.collection).count if self.exists() else 0

    def close(self) -> None:
        self.client.close()

    # Writes ---------------------------------------------------------------------
    def upsert(self, chunks: list[Chunk], dense: list[list[float]], sparse: list[SparseVec], batch_size: int = 128) -> None:
        if not (len(chunks) == len(dense) == len(sparse)):
            raise ValueError("chunks, dense vectors, and sparse vectors must align")
        points = [
            models.PointStruct(
                id=point_id(chunk.id),
                vector={
                    self.DENSE: dense_vector,
                    self.SPARSE: models.SparseVector(indices=sparse_vector.indices, values=sparse_vector.values),
                },
                payload=self._payload(chunk),
            )
            for chunk, dense_vector, sparse_vector in zip(chunks, dense, sparse)
        ]
        for start in range(0, len(points), batch_size):
            self.client.upsert(self.collection, points[start : start + batch_size])

    @staticmethod
    def _payload(chunk: Chunk) -> dict:
        payload = chunk.model_dump(mode="json")
        payload["decision_date_int"] = date_int(chunk.metadata.decision_date)
        payload["effective_date_int"] = date_int(chunk.metadata.effective_date)
        return payload

    # Reads ----------------------------------------------------------------------
    def texts_by_chunk_id(self, chunk_ids: list[str]) -> dict[str, str]:
        """Point lookup by id: document_id and order carry no payload index."""
        if not chunk_ids:
            return {}
        points = self.client.retrieve(self.collection, ids=[point_id(i) for i in chunk_ids], with_payload=["id", "text"])
        return {point.payload["id"]: point.payload["text"] for point in points}

    def search_dense(self, vector: list[float], k: int, filters: RetrievalFilters | None = None) -> list[tuple[Chunk, float]]:
        return self._query(vector, self.DENSE, k, filters)

    def search_sparse(self, vector: SparseVec, k: int, filters: RetrievalFilters | None = None) -> list[tuple[Chunk, float]]:
        query = models.SparseVector(indices=vector.indices, values=vector.values)
        return self._query(query, self.SPARSE, k, filters)

    def _query(self, query, using: str, k: int, filters: RetrievalFilters | None) -> list[tuple[Chunk, float]]:
        result = self.client.query_points(
            self.collection,
            query=query,
            using=using,
            limit=k,
            query_filter=self._to_filter(filters),
            with_payload=True,
        )
        hits: list[tuple[Chunk, float]] = []
        for point in result.points:
            payload = {key: value for key, value in point.payload.items() if key not in _DERIVED_KEYS}
            hits.append((Chunk.model_validate(payload), float(point.score)))
        return hits

    @staticmethod
    def _to_filter(filters: RetrievalFilters | None) -> models.Filter | None:
        conditions = to_conditions(filters)
        if not conditions:
            return None
        must: list[models.Condition] = []
        for condition in conditions:
            if isinstance(condition, MatchAny):
                must.append(models.FieldCondition(key=condition.key, match=models.MatchAny(any=condition.values)))
            elif isinstance(condition, Range):
                must.append(models.FieldCondition(key=condition.key, range=models.Range(gte=condition.gte, lte=condition.lte)))
        return models.Filter(must=must)

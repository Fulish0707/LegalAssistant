"""Dense embeddings served by Ollama's ``/api/embed`` endpoint."""

from __future__ import annotations

import httpx


class OllamaEmbedder:
    def __init__(self, model: str, base_url: str, batch_size: int = 16, timeout: float = 120.0,
                 transport: httpx.BaseTransport | None = None, query_num_ctx: int | None = None) -> None:
        self.model = model
        # Context window for query embeddings only. Ollama sizes bge-m3's compute buffer
        # to the window (1.17 GB at its default 4096), and a short query needs far less.
        self.query_num_ctx = query_num_ctx
        self.batch_size = max(1, batch_size)
        self.name = f"ollama:{model}"
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.embed_query("차원 확인"))
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._call(texts[start : start + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._call([text], num_ctx=self.query_num_ctx)[0]

    def _call(self, inputs: list[str], num_ctx: int | None = None) -> list[list[float]]:
        body: dict = {"model": self.model, "input": inputs, "truncate": False}
        if num_ctx:
            body["options"] = {"num_ctx": num_ctx}
        response = self._client.post("/api/embed", json=body)
        response.raise_for_status()
        embeddings = response.json()["embeddings"]
        if self._dimension is None and embeddings:
            self._dimension = len(embeddings[0])
        return embeddings

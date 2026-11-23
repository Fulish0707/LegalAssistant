"""Dense embeddings served by Ollama's ``/api/embed`` endpoint."""

from __future__ import annotations

import httpx


class OllamaEmbedder:
    def __init__(self, model: str, base_url: str, batch_size: int = 16, timeout: float = 120.0,
                 transport: httpx.BaseTransport | None = None, query_num_ctx: int | None = None,
                 query_num_gpu: int | None = None) -> None:
        self.model = model
        # Context window for query embeddings only. Ollama sizes bge-m3's compute buffer
        # to the window (1.17 GB at its default 4096), and a short query needs far less.
        self.query_num_ctx = query_num_ctx
        # GPU layers for query embeddings only; 0 runs bge-m3 on CPU (about +0.03 s per query) and leaves
        # the card to the answer model and the reranker. Must match the chat app's OLLAMA_EMBED_NUM_GPU
        # or Ollama reloads bge-m3 between the two clients.
        self.query_num_gpu = query_num_gpu
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
        return self._call([text], num_ctx=self.query_num_ctx, num_gpu=self.query_num_gpu)[0]

    def _call(self, inputs: list[str], num_ctx: int | None = None, num_gpu: int | None = None) -> list[list[float]]:
        body: dict = {"model": self.model, "input": inputs, "truncate": False}
        options = {key: value for key, value in (("num_ctx", num_ctx or None), ("num_gpu", num_gpu)) if value is not None}
        if options:
            body["options"] = options
        response = self._client.post("/api/embed", json=body)
        response.raise_for_status()
        embeddings = response.json()["embeddings"]
        if self._dimension is None and embeddings:
            self._dimension = len(embeddings[0])
        return embeddings

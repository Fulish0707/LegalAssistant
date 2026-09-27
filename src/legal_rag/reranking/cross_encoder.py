"""Rerank fused candidates with a cross-encoder; needs the optional ``rerank`` extra."""
from __future__ import annotations

from typing import Any, Callable, Protocol

from legal_rag.schemas import ScoredChunk


class _Encoder(Protocol):
    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> Any: ...


Loader = Callable[..., _Encoder]


def _load_cross_encoder(model: str, *, device: str, max_length: int, dtype: str) -> _Encoder:
    import torch
    from sentence_transformers import CrossEncoder  # heavy import, optional extra

    return CrossEncoder(model, device=device, max_length=max_length,
                        model_kwargs={'torch_dtype': getattr(torch, dtype)})


class CrossEncoderReranker:
    """Scores every (query, chunk text) pair and re-sorts the candidates by that score.

    The model is loaded on first use, so importing or constructing the reranker stays cheap and
    offline; ``loader`` replaces the load for tests. Scores are whatever the model emits —
    sentence-transformers applies a model's own activation, so bge-reranker-v2-m3 yields [0, 1]
    while another model may yield signed logits — and none are dropped whatever their sign.
    Ties keep the fused order, so a run is reproducible.
    """

    name = 'cross_encoder'

    def __init__(self, model: str, device: str = 'cpu', *, batch_size: int = 16, max_length: int = 512,
                 dtype: str = 'float32', loader: Loader | None = None) -> None:
        self.model = model
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_length = max_length
        self._loader = loader or _load_cross_encoder
        self._encoder: _Encoder | None = None

    @property
    def is_loaded(self) -> bool:
        return self._encoder is not None

    def encoder(self) -> _Encoder:
        if self._encoder is None:
            self._encoder = self._loader(self.model, device=self.device, max_length=self.max_length, dtype=self.dtype)
        return self._encoder

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        if k <= 0 or not candidates:
            return []
        pairs = [(query, candidate.chunk.text) for candidate in candidates]
        scores = [float(score) for score in self.encoder().predict(pairs, batch_size=self.batch_size)]
        order = sorted(range(len(candidates)), key=lambda index: (-scores[index], index))
        return [candidates[index].model_copy(update={'score': scores[index], 'retriever': 'reranked'})
                for index in order[:k]]

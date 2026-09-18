"""Cross-encoder reranking with a fake model loader: no download, no network."""
from __future__ import annotations

from legal_rag.config import Settings
from legal_rag.factory import build_reranker
from legal_rag.reranking.cross_encoder import CrossEncoderReranker
from legal_rag.schemas import ScoredChunk
from tests.legal_rag.unit.test_store import _chunks


class _FakeEncoder:
    """Returns a score per pair from a lookup on the chunk text."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[dict] = []

    def predict(self, pairs, **kwargs):
        self.calls.append({'pairs': list(pairs), **kwargs})
        return [self.scores.get(text, 0.0) for _, text in pairs]


class _FakeLoader:
    def __init__(self, encoder: _FakeEncoder) -> None:
        self.encoder = encoder
        self.calls: list[dict] = []

    def __call__(self, model, *, device, max_length):
        self.calls.append({'model': model, 'device': device, 'max_length': max_length})
        return self.encoder


def _candidates():
    chunks = _chunks()
    return [ScoredChunk(chunk=chunk, score=1.0 - index / 10, retriever='fused')
            for index, chunk in enumerate(chunks)]


def _reranker(scores, **kwargs):
    loader = _FakeLoader(_FakeEncoder(scores))
    return CrossEncoderReranker(model='fake/model', device='cpu', loader=loader, **kwargs), loader


def test_name_and_constructor_does_not_load_the_model():
    reranker, loader = _reranker({})
    assert reranker.name == 'cross_encoder'
    assert loader.calls == []


def test_reorders_candidates_by_cross_encoder_score():
    candidates = _candidates()
    scores = {candidates[0].chunk.text: -2.0, candidates[1].chunk.text: 5.0, candidates[2].chunk.text: 1.0}
    reranker, loader = _reranker(scores)
    hits = reranker.rerank('보증금 반환', candidates, k=3)
    assert [hit.chunk.id for hit in hits] == [candidates[1].chunk.id, candidates[2].chunk.id, candidates[0].chunk.id]
    assert [hit.score for hit in hits] == [5.0, 1.0, -2.0]
    assert {hit.retriever for hit in hits} == {'reranked'}
    assert loader.calls == [{'model': 'fake/model', 'device': 'cpu', 'max_length': 512}]


def test_negative_scores_are_kept_rather_than_dropped():
    candidates = _candidates()
    reranker, _ = _reranker({chunk.chunk.text: -3.0 for chunk in candidates})
    assert len(reranker.rerank('q', candidates, k=3)) == 3


def test_truncates_to_k_and_scores_every_candidate_first():
    candidates = _candidates()
    scores = {candidates[2].chunk.text: 9.0}
    reranker, loader = _reranker(scores)
    hits = reranker.rerank('q', candidates, k=1)
    assert [hit.chunk.id for hit in hits] == [candidates[2].chunk.id]
    assert len(loader.encoder.calls[0]['pairs']) == 3


def test_ties_keep_the_fused_order():
    candidates = _candidates()
    reranker, _ = _reranker({chunk.chunk.text: 1.0 for chunk in candidates})
    hits = reranker.rerank('q', candidates, k=3)
    assert [hit.chunk.id for hit in hits] == [chunk.chunk.id for chunk in candidates]


def test_empty_candidates_and_non_positive_k_return_nothing_and_never_load():
    reranker, loader = _reranker({})
    assert reranker.rerank('q', [], k=5) == []
    assert reranker.rerank('q', _candidates(), k=0) == []
    assert reranker.rerank('q', _candidates(), k=-1) == []
    assert loader.calls == []


def test_batch_size_and_max_length_reach_the_model():
    candidates = _candidates()
    reranker, loader = _reranker({}, batch_size=4, max_length=128)
    reranker.rerank('q', candidates, k=2)
    assert loader.calls[0]['max_length'] == 128
    assert loader.encoder.calls[0]['batch_size'] == 4


def test_the_model_is_loaded_once_and_reused():
    candidates = _candidates()
    reranker, loader = _reranker({})
    reranker.rerank('q', candidates, k=2)
    reranker.rerank('q', candidates, k=2)
    assert len(loader.calls) == 1


def test_query_is_paired_with_each_candidate_text():
    candidates = _candidates()
    reranker, loader = _reranker({})
    reranker.rerank('특허 침해', candidates, k=3)
    assert loader.encoder.calls[0]['pairs'] == [('특허 침해', chunk.chunk.text) for chunk in candidates]


def test_factory_builds_a_lazy_cross_encoder_reranker(tmp_path):
    settings = Settings(paths={'root': str(tmp_path)},
                        rerank={'backend': 'cross_encoder', 'model': 'BAAI/bge-reranker-v2-m3', 'device': 'cuda'})
    reranker = build_reranker(settings)
    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.name == 'cross_encoder'
    assert (reranker.model, reranker.device) == ('BAAI/bge-reranker-v2-m3', 'cuda')
    assert not reranker.is_loaded


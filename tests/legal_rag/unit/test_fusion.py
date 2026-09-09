from legal_rag.reranking import NoopReranker
from legal_rag.retrieval.fusion import RRFFuser
from legal_rag.retrieval.hybrid import HybridRetriever
from legal_rag.schemas import ScoredChunk
from tests.legal_rag.unit.test_store import _chunks


def _sc(chunk, score, name):
    return ScoredChunk(chunk=chunk, score=score, retriever=name)


def test_rrf_rewards_chunks_present_in_both_lists():
    c1, c2, c3 = _chunks()
    dense = [_sc(c1, 0.9, "dense"), _sc(c2, 0.8, "dense")]
    sparse = [_sc(c3, 5.0, "sparse"), _sc(c1, 4.0, "sparse")]
    fused = RRFFuser(rrf_k=60).fuse([dense, sparse], k=3)
    assert fused[0].chunk.id == c1.id and fused[0].retriever == "fused"
    assert abs(fused[0].score - (1 / 61 + 1 / 62)) < 1e-9
    assert fused[0].chunk == c1 and len(fused) == 3


def test_rrf_applies_leg_weights():
    c1, c2, _ = _chunks()
    fused = RRFFuser(rrf_k=60, weights={"dense": 0.0, "sparse": 1.0}).fuse([[_sc(c1, 1, "dense")], [_sc(c2, 1, "sparse")]], k=2)
    assert [h.chunk.id for h in fused] == [c2.id]


class _Stub:
    def __init__(self, name, hits):
        self.name, self.hits = name, hits

    def retrieve(self, query, k, filters=None):
        return self.hits[:k]


def test_hybrid_runs_every_query_through_every_leg_and_reranks():
    c1, c2, c3 = _chunks()
    dense = _Stub("dense", [_sc(c1, 0.9, "dense")])
    sparse = _Stub("sparse", [_sc(c2, 1.0, "sparse")])
    hybrid = HybridRetriever([dense, sparse], RRFFuser(60), NoopReranker(), k_leg=5, k_fused=5)
    hits = hybrid.retrieve_many(["q1", "q2"], k=2)
    assert {h.chunk.id for h in hits} == {c1.id, c2.id} and all(h.retriever == "reranked" for h in hits)
    assert set(hybrid.last_trace) == {"dense:q1", "sparse:q1", "dense:q2", "sparse:q2", "fused", "reranked"}
    assert hybrid.retrieve("q1", k=1)[0].chunk.id in {c1.id, c2.id}

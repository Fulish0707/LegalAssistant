from legal_rag.reranking import NoopReranker
from legal_rag.retrieval.fusion import RRFFuser
from legal_rag.retrieval.hybrid import HybridRetriever
from legal_rag.schemas import Chunk, ChunkMetadata, DocumentType, ScoredChunk
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


# --- leg floor --------------------------------------------------------------------------


def _hit(chunk_id, retriever):
    chunk = Chunk(id=chunk_id, document_id=chunk_id, document_type=DocumentType.court_decision,
                  path="p", order=0, text="t", title="t", metadata=ChunkMetadata(dataset="d", corpus_version="v"))
    return ScoredChunk(chunk=chunk, score=1.0, retriever=retriever)


def _agreeing_and_lonely_legs():
    """Corpus a's legs agree on a1..a6; corpus b's sparse leg alone holds b-top.

    Without a floor, every chunk both of a's legs return outscores b-top, the rank-1 hit of
    its list, so fusion never hands b-top to the reranker."""
    a_dense = [_hit(f"a{i}", "a:dense") for i in range(1, 7)]
    a_sparse = [_hit(f"a{i}", "a:sparse") for i in range(1, 7)]
    b_sparse = [_hit("b-top", "b:sparse"), _hit("b-2", "b:sparse")]
    return [a_dense, a_sparse, b_sparse]


def test_without_a_floor_a_lone_rank_one_hit_loses_to_leg_agreement():
    fused = RRFFuser(rrf_k=60).fuse(_agreeing_and_lonely_legs(), k=4)
    assert [h.chunk.id for h in fused] == ["a1", "a2", "a3", "a4"]


def test_leg_floor_admits_every_lists_top_hits():
    fused = RRFFuser(rrf_k=60, leg_floor=1).fuse(_agreeing_and_lonely_legs(), k=4)
    assert {h.chunk.id for h in fused} == {"a1", "a2", "a3", "b-top"}


def test_leg_floor_output_stays_ordered_by_fused_score():
    fused = RRFFuser(rrf_k=60, leg_floor=1).fuse(_agreeing_and_lonely_legs(), k=4)
    assert [h.chunk.id for h in fused] == ["a1", "a2", "a3", "b-top"]
    assert all(x.score >= y.score for x, y in zip(fused, fused[1:]))


def test_leg_floor_backfills_by_fused_score_so_k_is_filled():
    fused = RRFFuser(rrf_k=60, leg_floor=2).fuse(_agreeing_and_lonely_legs(), k=8)
    assert [h.chunk.id for h in fused] == ["a1", "a2", "a3", "a4", "a5", "a6", "b-top", "b-2"]


def test_leg_floor_admits_rank_by_rank_across_lists():
    lists = [[_hit(f"{leg}{i}", f"{leg}:dense") for i in range(3)] for leg in "xyz"]
    fused = RRFFuser(rrf_k=60, leg_floor=2).fuse(lists, k=6)
    assert {h.chunk.id for h in fused} == {"x0", "y0", "z0", "x1", "y1", "z1"}


def test_leg_floor_fills_at_most_half_of_k():
    """A multi-query chat search fuses queries x legs lists (16-32); a floor of 3 on each would
    fill k=50 almost entirely by round-robin. Half the slots always stay for fused score."""
    singles = [[_hit(f"u{i}_{r}", f"c{i}:dense") for r in range(3)] for i in range(16)]
    agreeing = [[_hit(f"A{j}", f"z:{leg}") for j in range(20)] for leg in ("dense", "sparse")]
    fused = RRFFuser(rrf_k=60, leg_floor=3).fuse(singles + agreeing, k=20)
    assert sum(h.chunk.id.startswith("A") for h in fused) >= 10
    assert sum(h.chunk.id.startswith("u") for h in fused) == 10
def test_leg_floor_without_the_floor_mattering_matches_plain_rrf():
    c1, c2, c3 = _chunks()
    dense = [_sc(c1, 0.9, "dense"), _sc(c2, 0.8, "dense")]
    sparse = [_sc(c3, 5.0, "sparse"), _sc(c1, 4.0, "sparse")]
    plain = RRFFuser(rrf_k=60).fuse([dense, sparse], k=3)
    floored = RRFFuser(rrf_k=60, leg_floor=2).fuse([dense, sparse], k=3)
    assert [(h.chunk.id, h.score) for h in floored] == [(h.chunk.id, h.score) for h in plain]


def test_leg_floor_gives_a_zero_weighted_leg_nothing():
    lists = [[_hit("a1", "a:dense"), _hit("a2", "a:dense")], [_hit("b1", "b:dense")]]
    fused = RRFFuser(rrf_k=60, weights={"b:dense": 0.0}, leg_floor=1).fuse(lists, k=2)
    assert [h.chunk.id for h in fused] == ["a1", "a2"]


def test_leg_floor_applies_to_each_query_list_of_a_multi_query_search():
    # retrieve_many hands the fuser one list per (query, leg); each is its own list here too.
    q1 = [[_hit(f"a{i}", "a:dense") for i in range(1, 7)], [_hit(f"a{i}", "a:sparse") for i in range(1, 7)]]
    q2 = [[_hit("a1", "a:dense")], [_hit("rewrite-top", "a:sparse")]]
    assert "rewrite-top" not in {h.chunk.id for h in RRFFuser(rrf_k=60).fuse(q1 + q2, k=4)}
    fused = RRFFuser(rrf_k=60, leg_floor=1).fuse(q1 + q2, k=4)
    assert {h.chunk.id for h in fused} == {"a1", "a2", "a3", "rewrite-top"}

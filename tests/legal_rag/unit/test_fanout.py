import json

import pytest

from legal_rag.config import load_settings
from legal_rag.factory import build_corpus_legs
from legal_rag.ingestion import import_processed as importer
from legal_rag.retrieval.index_builder import build_index


def settings(tmp_path, version="a", **retrieval):
    overrides = {"paths": {"root": str(tmp_path)}, "corpus": {"version": version},
                 "embedding": {"backend": "fake"}}
    if retrieval:
        overrides["retrieval"] = retrieval
    return load_settings(profile="default", overrides=overrides)


def make_corpus(tmp_path, version, terms):
    """Index one tiny corpus whose vocabulary is `terms`, so BM25 df differs per corpus."""
    source = tmp_path / f"src-{version}"
    source.mkdir()
    docs = [dict(document_id=f"{version}-doc{i}", document_type="statute",
                 legal_domain=version, title=f"{version} 법", text=f"{term} 조문 내용",
                 metadata={}) for i, term in enumerate(terms)]
    chunks = [dict(d, chunk_id=f"{version}-chunk{i}", chunk_index=0, total_chunks=1)
              for i, d in enumerate(docs)]
    for name, rows in (("documents.jsonl", docs), ("chunks.jsonl", chunks)):
        (source / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    cfg = settings(tmp_path, version)
    importer.import_processed(source, cfg)
    build_index(cfg)
    return cfg


def test_legs_are_named_after_their_corpus(tmp_path):
    make_corpus(tmp_path, "alpha", ["특허", "상표"])
    make_corpus(tmp_path, "beta", ["절도", "사기"])
    cfg = settings(tmp_path, "alpha", corpora=["alpha", "beta"])
    legs = build_corpus_legs(cfg)
    assert [leg.name for leg in legs] == [
        "alpha:dense", "alpha:sparse", "beta:dense", "beta:sparse"]


def test_each_corpus_sparse_leg_uses_its_own_statistics(tmp_path):
    # The point of per-corpus BM25: a term only beta knows must still retrieve beta hits,
    # and must not be silently dropped because alpha's df lacks it.
    make_corpus(tmp_path, "alpha", ["특허", "상표"])
    make_corpus(tmp_path, "beta", ["절도", "사기"])
    cfg = settings(tmp_path, "alpha", corpora=["alpha", "beta"])
    legs = {leg.name: leg for leg in build_corpus_legs(cfg)}
    assert legs["beta:sparse"].retrieve("절도", 5)
    assert not legs["alpha:sparse"].retrieve("절도", 5)


def test_hits_carry_the_corpus_qualified_retriever_name(tmp_path):
    make_corpus(tmp_path, "alpha", ["특허", "상표"])
    cfg = settings(tmp_path, "alpha", corpora=["alpha"])
    legs = {leg.name: leg for leg in build_corpus_legs(cfg)}
    hits = legs["alpha:sparse"].retrieve("특허", 5)
    assert hits and all(hit.retriever == "alpha:sparse" for hit in hits)


def test_a_corpus_indexed_with_another_embedder_is_named_in_the_error(tmp_path):
    from legal_rag.factory import IndexCompatibilityError
    make_corpus(tmp_path, "alpha", ["특허"])
    make_corpus(tmp_path, "beta", ["절도"])
    index = settings(tmp_path, "beta").index_corpus_dir() / "index.json"
    metadata = json.loads(index.read_text())
    metadata["embedder"] = "bge-m3:1024"
    index.write_text(json.dumps(metadata), encoding="utf-8")
    cfg = settings(tmp_path, "alpha", corpora=["alpha", "beta"])
    with pytest.raises(IndexCompatibilityError) as excinfo:
        build_corpus_legs(cfg)
    assert "beta" in str(excinfo.value) and "embedder" in str(excinfo.value)


def test_leg_weights_multiply_corpus_and_retriever_weights(tmp_path):
    from legal_rag.factory import fusion_weights
    cfg = settings(tmp_path, "alpha", corpora=["alpha", "beta"],
                   corpus_weights={"beta": 0.5}, dense_weight=1.0, sparse_weight=2.0)
    weights = fusion_weights(cfg)
    assert weights == {"alpha:dense": 1.0, "alpha:sparse": 2.0,
                       "beta:dense": 0.5, "beta:sparse": 1.0}


def test_a_down_weighted_corpus_ranks_below_a_full_weight_one(tmp_path):
    from legal_rag.retrieval.fusion import RRFFuser
    from legal_rag.schemas import Chunk, ChunkMetadata, DocumentType, ScoredChunk

    def hit(chunk_id, retriever):
        chunk = Chunk(id=chunk_id, document_id=chunk_id, document_type=DocumentType.statute,
                      path="p", order=0, text="t", title="t",
                      metadata=ChunkMetadata(dataset="d", corpus_version="v"))
        return ScoredChunk(chunk=chunk, score=1.0, retriever=retriever)

    fuser = RRFFuser(rrf_k=60, weights={"alpha:dense": 1.0, "beta:dense": 0.2})
    fused = fuser.fuse([[hit("beta-top", "beta:dense")],
                        [hit("alpha-1", "alpha:dense"), hit("alpha-2", "alpha:dense")]], k=3)
    assert [h.chunk.id for h in fused] == ["alpha-1", "alpha-2", "beta-top"]


def test_configured_corpora_produce_a_retriever_spanning_both(tmp_path):
    from legal_rag.factory import build_hybrid_retriever
    make_corpus(tmp_path, "alpha", ["특허", "상표"])
    make_corpus(tmp_path, "beta", ["절도", "사기"])
    cfg = settings(tmp_path, "alpha", corpora=["alpha", "beta"])
    retriever = build_hybrid_retriever(cfg)
    assert [leg.name for leg in retriever.legs] == [
        "alpha:dense", "alpha:sparse", "beta:dense", "beta:sparse"]
    hits = retriever.retrieve("절도", k=5)
    assert any(hit.chunk.id.startswith("beta") for hit in hits)


def test_no_configured_corpora_keeps_single_corpus_behaviour(tmp_path):
    from legal_rag.factory import build_hybrid_retriever
    make_corpus(tmp_path, "alpha", ["특허"])
    retriever = build_hybrid_retriever(settings(tmp_path, "alpha"))
    assert [leg.name for leg in retriever.legs] == ["alpha:dense", "alpha:sparse"]

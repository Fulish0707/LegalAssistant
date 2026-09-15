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

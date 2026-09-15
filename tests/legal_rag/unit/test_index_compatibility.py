import json

import pytest

from legal_rag.config import load_settings
from legal_rag.factory import IndexCompatibilityError, build_hybrid_retriever, check_index_compatibility


def settings(tmp_path, **sparse):
    return load_settings(profile="default", overrides={
        "paths": {"root": str(tmp_path)},
        "corpus": {"version": "v1"},
        "embedding": {"backend": "fake"},
        "sparse": sparse or {"tokenizer": "regex"},
    })


def write_index(tmp_path, **metadata):
    cfg = settings(tmp_path)
    index_dir = cfg.index_corpus_dir()
    index_dir.mkdir(parents=True, exist_ok=True)
    base = {"collection": "c", "embedder": "fake:64", "dimension": 64, "tokenizer": "regex",
            "chunks": 1, "corpus_version": "v1", "config_hash": "x", "qdrant_mode": "embedded"}
    base.update(metadata)
    (index_dir / "index.json").write_text(json.dumps(base), encoding="utf-8")
    (index_dir / "bm25.json").write_text(
        json.dumps({"df": {"a": 1}, "n_docs": 1, "avg_len": 1.0, "k1": 1.2, "b": 0.75}), encoding="utf-8")
    return cfg


def test_matching_tokenizer_returns_metadata(tmp_path):
    write_index(tmp_path)
    metadata = check_index_compatibility(settings(tmp_path, tokenizer="regex"))
    assert metadata["tokenizer"] == "regex"


def test_tokenizer_mismatch_fails_loudly(tmp_path):
    # Querying with a different analyzer than the index was built with makes every
    # query term miss the document-frequency map, so the sparse leg silently dies.
    write_index(tmp_path, tokenizer="regex")
    with pytest.raises(IndexCompatibilityError) as excinfo:
        check_index_compatibility(settings(tmp_path, tokenizer="kiwi"))
    message = str(excinfo.value)
    assert "tokenizer" in message and "regex" in message and "kiwi" in message
    assert "build_index" in message


def test_index_built_before_tokenizer_was_recorded_fails(tmp_path):
    cfg = write_index(tmp_path)
    metadata = json.loads((cfg.index_corpus_dir() / "index.json").read_text())
    del metadata["tokenizer"]
    (cfg.index_corpus_dir() / "index.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(IndexCompatibilityError, match="tokenizer"):
        check_index_compatibility(settings(tmp_path, tokenizer="regex"))


def test_missing_bm25_statistics_fail(tmp_path):
    cfg = write_index(tmp_path)
    (cfg.index_corpus_dir() / "bm25.json").unlink()
    with pytest.raises(IndexCompatibilityError, match="bm25.json"):
        check_index_compatibility(settings(tmp_path, tokenizer="regex"))


def test_missing_index_reports_where_it_looked(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_index"):
        check_index_compatibility(settings(tmp_path, tokenizer="regex"))


def test_embedder_mismatch_fails(tmp_path):
    write_index(tmp_path, embedder="bge-m3:1024")
    with pytest.raises(IndexCompatibilityError, match="embedder"):
        check_index_compatibility(settings(tmp_path, tokenizer="regex"), embedder_name="fake:64")


def test_every_mismatch_is_reported_at_once(tmp_path):
    write_index(tmp_path, tokenizer="regex", embedder="bge-m3:1024")
    with pytest.raises(IndexCompatibilityError) as excinfo:
        check_index_compatibility(settings(tmp_path, tokenizer="kiwi"), embedder_name="fake:64")
    message = str(excinfo.value)
    assert "tokenizer" in message and "embedder" in message


def test_hybrid_retriever_refuses_an_incompatible_index(tmp_path):
    # The guard has to sit on the retrieval path itself, not merely be available.
    write_index(tmp_path)
    with pytest.raises(IndexCompatibilityError, match="tokenizer"):
        build_hybrid_retriever(settings(tmp_path, tokenizer="kiwi"))

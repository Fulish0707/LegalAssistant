from pathlib import Path

import pytest

from legal_rag.config import Settings, load_settings


def test_default_profile_loads_with_expected_defaults():
    settings = load_settings(profile="default")
    assert isinstance(settings, Settings)
    assert settings.embedding.backend == "ollama"
    assert settings.embedding.model == "bge-m3"
    assert settings.retrieval.k_final == 5
    assert settings.qdrant.mode == "embedded"


def test_profile_overrides_default(tmp_path: Path):
    (tmp_path / "default.yaml").write_text("retrieval:\n  k_final: 5\n", encoding="utf-8")
    (tmp_path / "fast.yaml").write_text("retrieval:\n  k_final: 3\n", encoding="utf-8")
    settings = load_settings(profile="fast", config_dir=tmp_path)
    assert settings.retrieval.k_final == 3


def test_environment_overrides_yaml(monkeypatch, tmp_path: Path):
    (tmp_path / "default.yaml").write_text("retrieval:\n  k_final: 5\n", encoding="utf-8")
    monkeypatch.setenv("LEGAL_RAG__RETRIEVAL__K_FINAL", "7")
    settings = load_settings(profile="default", config_dir=tmp_path)
    assert settings.retrieval.k_final == 7


def test_overrides_beat_yaml(tmp_path: Path):
    (tmp_path / "default.yaml").write_text("corpus:\n  version: v1\n", encoding="utf-8")
    settings = load_settings(profile="default", config_dir=tmp_path, overrides={"corpus": {"version": "x"}})
    assert settings.corpus.version == "x"


def test_secrets_in_yaml_are_rejected(tmp_path: Path):
    (tmp_path / "default.yaml").write_text("qdrant:\n  api_key: leaked\n", encoding="utf-8")
    with pytest.raises(ValueError, match="api_key"):
        load_settings(profile="default", config_dir=tmp_path)


def test_hash_is_stable_and_ignores_secrets(monkeypatch):
    a = load_settings(profile="default").hash()
    monkeypatch.setenv("LEGAL_RAG__QDRANT__API_KEY", "secret")
    b = load_settings(profile="default").hash()
    assert a == b


def test_derived_paths(tmp_path: Path):
    settings = load_settings(profile="default", overrides={"paths": {"root": str(tmp_path)}, "corpus": {"version": "c"}})
    assert settings.processed_corpus_dir() == tmp_path / "data" / "processed" / "c"
    assert settings.qdrant_path() == tmp_path / "data" / "indexes" / "c" / "qdrant"
    assert settings.raw_dir() == tmp_path / "data" / "raw"

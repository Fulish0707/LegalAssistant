import json
from pathlib import Path

from legal_rag.config import load_settings
from legal_rag.ingestion.corpus import Corpus
from legal_rag.ingestion.pipeline import run_ingestion

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "raw"


def _settings(tmp_path: Path):
    return load_settings(profile="default", overrides={
        "paths": {"root": str(tmp_path), "raw_dir": str(FIXTURES)},
        "corpus": {"version": "test"},
    })


def test_pipeline_writes_documents_chunks_and_manifest(tmp_path: Path):
    report = run_ingestion(_settings(tmp_path))
    out = tmp_path / "data" / "processed" / "test"
    assert (out / "documents.jsonl").exists() and (out / "chunks.jsonl").exists()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["corpus_version"] == "test"
    assert manifest["counts"]["normalized"] == 5
    assert manifest["counts"]["unrecognized"] == 1
    assert manifest["dropped"][0]["stage"] == "detect"
    assert any(f["file"].endswith("judgment_001.json") for f in manifest["raw_files"])
    assert report.documents == 5 and report.chunks > 5


def test_pipeline_is_byte_reproducible(tmp_path: Path):
    run_ingestion(_settings(tmp_path / "a"))
    run_ingestion(_settings(tmp_path / "b"))
    for name in ("documents.jsonl", "chunks.jsonl"):
        assert (tmp_path / "a/data/processed/test" / name).read_bytes() == (tmp_path / "b/data/processed/test" / name).read_bytes()


def test_duplicates_are_dropped(tmp_path: Path):
    raw = tmp_path / "raw" / "statutes"
    raw.mkdir(parents=True)
    src = (FIXTURES / "statutes" / "civil_act_excerpt.json").read_text(encoding="utf-8")
    (raw / "a.json").write_text(src, encoding="utf-8")
    (raw / "b.json").write_text(src, encoding="utf-8")
    settings = load_settings(profile="default", overrides={"paths": {"root": str(tmp_path), "raw_dir": str(tmp_path / "raw")}, "corpus": {"version": "t"}})
    report = run_ingestion(settings)
    assert report.documents == 1 and report.deduplicated == 1


def test_corpus_reader_round_trips(tmp_path: Path):
    run_ingestion(_settings(tmp_path))
    corpus = Corpus.load(tmp_path / "data" / "processed" / "test")
    docs = list(corpus.documents())
    chunks = list(corpus.chunks())
    assert len(docs) == 5 and all(c.document_id in {d.id for d in docs} for c in chunks)
    assert corpus.document(chunks[0].document_id).id == chunks[0].document_id
    assert corpus.manifest().counts.documents == 5

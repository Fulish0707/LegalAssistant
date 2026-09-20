"""Re-deriving a corpus's chunks from its documents."""

from __future__ import annotations

import json

import pytest

from legal_rag.chunking.imported import ChunkingOptions
from legal_rag.ingestion.rechunk import rechunk_corpus

STATUTE = "제1조(목적)\n이 법은 상표를 보호한다.\n제2조(정의)\n" + ("정의 본문이다. " * 300) + "\n제3조(적용)\n적용한다.\n"
RULING = (
    "이 사건 심판청구를 기각한다.\n1. 기초사실\n가. 절차의 경위\n(1) 특허출원 : 2008. 12. 31.\n"
    "나. 청구범위\n" + ("청구범위의 내용이다. " * 120) + "\n2. 판단\n가. 진보성\n진보성이 없다.\n"
)
FLAT = "구조가 없는 본문이다. " * 300


def _doc(identifier: str, kind: str, text: str, title: str) -> dict:
    return {
        "id": identifier,
        "document_type": kind,
        "title": title,
        "text": text,
        "jurisdiction": "KR",
        "legal_domain": "intellectual_property",
        "source": {"dataset": "aihub_preprocessed", "file": "documents.jsonl", "record_key": identifier},
        "metadata": {"original": {"document_id": identifier, "authority": "특허심판원", "metadata": {}}},
        "format": "external_processed_v1",
    }


@pytest.fixture
def corpus(tmp_path):
    source = tmp_path / "ip-v1"
    source.mkdir()
    rows = [
        _doc("s1", "statute", STATUTE, "상표법"),
        _doc("p1", "patent_trial_decision", RULING, "발명의 명칭"),
        _doc("j1", "judgment", FLAT, "상표법위반"),
    ]
    (source / "documents.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    (source / "manifest.json").write_text(json.dumps({"corpus_version": "ip-v1", "counts": {}}), encoding="utf-8")
    return source


def _chunks(directory):
    return [json.loads(line) for line in (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]


class TestOutputs:
    def test_writes_the_three_files(self, corpus, tmp_path):
        out = tmp_path / "ip-v2"
        rechunk_corpus(corpus, out, corpus_version="ip-v2")
        assert {p.name for p in out.iterdir()} == {"documents.jsonl", "chunks.jsonl", "manifest.json", "report.json"}

    def test_documents_are_copied_byte_for_byte(self, corpus, tmp_path):
        out = tmp_path / "ip-v2"
        rechunk_corpus(corpus, out, corpus_version="ip-v2")
        assert (out / "documents.jsonl").read_bytes() == (corpus / "documents.jsonl").read_bytes()

    def test_refuses_an_existing_output(self, corpus, tmp_path):
        out = tmp_path / "ip-v2"
        out.mkdir()
        with pytest.raises(FileExistsError):
            rechunk_corpus(corpus, out, corpus_version="ip-v2")

    def test_leaves_nothing_behind_when_it_fails(self, corpus, tmp_path):
        (corpus / "documents.jsonl").write_text("{not json}\n", encoding="utf-8")
        out = tmp_path / "ip-v2"
        with pytest.raises(ValueError):
            rechunk_corpus(corpus, out, corpus_version="ip-v2")
        assert not out.exists()

    def test_manifest_records_the_new_version_and_counts(self, corpus, tmp_path):
        out = tmp_path / "ip-v2"
        report = rechunk_corpus(corpus, out, corpus_version="ip-v2")
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["corpus_version"] == "ip-v2"
        assert manifest["pipeline_version"] == "structure_rechunk_v1"
        assert manifest["counts"]["documents"] == 3
        assert manifest["counts"]["chunks"] == report["chunks"]


class TestChunks:
    def test_every_document_produces_chunks(self, corpus, tmp_path):
        out = tmp_path / "ip-v2"
        rechunk_corpus(corpus, out, corpus_version="ip-v2")
        assert {c["document_id"] for c in _chunks(out)} == {"s1", "p1", "j1"}

    def test_chunk_ids_are_unique(self, corpus, tmp_path):
        out = tmp_path / "ip-v2"
        rechunk_corpus(corpus, out, corpus_version="ip-v2")
        ids = [c["id"] for c in _chunks(out)]
        assert len(set(ids)) == len(ids)

    def test_chunks_carry_the_new_corpus_version(self, corpus, tmp_path):
        out = tmp_path / "ip-v2"
        rechunk_corpus(corpus, out, corpus_version="ip-v2")
        assert {c["metadata"]["corpus_version"] for c in _chunks(out)} == {"ip-v2"}

    def test_order_restarts_per_document(self, corpus, tmp_path):
        out = tmp_path / "ip-v2"
        rechunk_corpus(corpus, out, corpus_version="ip-v2")
        per_doc: dict[str, list[int]] = {}
        for chunk in _chunks(out):
            per_doc.setdefault(chunk["document_id"], []).append(chunk["order"])
        assert all(orders == list(range(len(orders))) for orders in per_doc.values())


class TestReport:
    def test_counts_parse_failures_by_type(self, corpus, tmp_path):
        report = rechunk_corpus(corpus, tmp_path / "ip-v2", corpus_version="ip-v2")
        assert report["parsed"]["statute"] == 1
        assert report["parsed"]["patent_trial_decision"] == 1
        assert report["fallback"]["judgment"] == 1
        assert report["documents"] == 3

    def test_reports_length_percentiles(self, corpus, tmp_path):
        report = rechunk_corpus(corpus, tmp_path / "ip-v2", corpus_version="ip-v2")
        lengths = report["chunk_chars"]
        assert lengths["median"] <= lengths["p90"] <= lengths["max"]
        assert lengths["max"] <= ChunkingOptions().max_chars

    def test_reports_the_heading_share(self, corpus, tmp_path):
        report = rechunk_corpus(corpus, tmp_path / "ip-v2", corpus_version="ip-v2")
        assert report["header_line_share"] == 1.0
        assert 0.0 < report["structural_heading_share"] < 1.0

    def test_keeps_examples_per_type(self, corpus, tmp_path):
        report = rechunk_corpus(corpus, tmp_path / "ip-v2", corpus_version="ip-v2", examples=3)
        assert len(report["examples"]["statute"]) == 3
        assert all(len(e) <= 200 for e in report["examples"]["statute"])


class TestLimit:
    def test_limit_stops_after_n_documents(self, corpus, tmp_path):
        out = tmp_path / "ip-v2"
        report = rechunk_corpus(corpus, out, corpus_version="ip-v2", limit=1)
        assert report["documents"] == 1
        assert {c["document_id"] for c in _chunks(out)} == {"s1"}
        assert len((out / "documents.jsonl").read_text(encoding="utf-8").splitlines()) == 1

    def test_limit_must_be_positive(self, corpus, tmp_path):
        with pytest.raises(ValueError):
            rechunk_corpus(corpus, tmp_path / "ip-v2", corpus_version="ip-v2", limit=0)

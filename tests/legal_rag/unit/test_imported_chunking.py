"""Structure-aware chunking of imported documents."""

from __future__ import annotations

import re
from collections import Counter

import pytest

from legal_rag.chunking.imported import ChunkingOptions, chunk_imported_document
from legal_rag.schemas import DocumentType, SourceRef
from legal_rag.schemas.documents import ImportedDocument

STATUTE_TEXT = "제1조(목적)\n짧은 목적 조항이다.\n" + "제2조(정의)\n" + ("정의 조항 본문이다. " * 200) + "\n제3조(적용)\n적용한다.\n"

DECISION_TEXT = (
    "이 사건 심판청구를 기각한다.\n"
    "1. 기초사실\n가. 절차의 경위\n(1) 특허출원 : 2008. 12. 31.\n(2) 거절결정 : 2011. 03. 30.\n"
    "나. 청구범위\n" + ("청구범위의 내용이다. " * 150) + "\n"
    "2. 판단\n가. 진보성\n진보성이 인정되지 아니한다.\n"
)

UNSTRUCTURED = "구조가 없는 문단이다. " * 400


def _document(text: str, kind: str, **over) -> ImportedDocument:
    return ImportedDocument(
        id=over.get("id", "doc-1"),
        document_type=DocumentType(kind),
        legal_domain="intellectual_property",
        title=over.get("title", "상표법"),
        text=text,
        source=SourceRef(dataset="aihub_preprocessed", file="documents.jsonl", record_key="doc-1"),
        metadata={"original": over.get("original", {"authority": None, "title": "상표법", "metadata": {}})},
    )


OPTS = ChunkingOptions(target_chars=1200, max_chars=2000, merge_threshold=400)


class TestHeaderLine:
    def test_every_chunk_starts_with_a_heading_line(self):
        doc = _document(DECISION_TEXT, "patent_trial_decision", id="2011101003196", title="게시 시스템")
        chunks, report = chunk_imported_document(doc, "ip-v2", OPTS)
        assert report.parsed
        for chunk in chunks:
            first = chunk.text.split("\n", 1)[0]
            assert first.startswith("2011101003196")
            assert len(first) <= 200

    def test_statute_header_names_law_and_article(self):
        doc = _document(STATUTE_TEXT, "statute")
        chunks, _ = chunk_imported_document(doc, "ip-v2", OPTS)
        assert chunks[0].text.split("\n")[0] == "상표법 제1조(목적)"

    def test_decision_header_shows_the_section_path(self):
        doc = _document(DECISION_TEXT, "patent_trial_decision", id="2011101003196", title="게시 시스템")
        chunks, _ = chunk_imported_document(doc, "ip-v2", OPTS)
        headers = [c.text.split("\n")[0] for c in chunks]
        assert any("1. 기초사실 > 나. 청구범위" in h for h in headers)
        assert any(" · " in h for h in headers)


class TestSizing:
    def test_no_chunk_exceeds_the_hard_maximum(self):
        for text, kind in ((STATUTE_TEXT, "statute"), (DECISION_TEXT, "judgment"), (UNSTRUCTURED, "judgment")):
            chunks, _ = chunk_imported_document(_document(text, kind), "ip-v2", OPTS)
            assert chunks and all(len(c.text) <= OPTS.max_chars for c in chunks)

    def test_small_sibling_sections_merge_under_their_parent(self):
        doc = _document(DECISION_TEXT, "patent_trial_decision")
        chunks, _ = chunk_imported_document(doc, "ip-v2", OPTS)
        merged = [c for c in chunks if "특허출원" in c.text and "거절결정" in c.text]
        assert len(merged) == 1
        assert "가. 절차의 경위" in merged[0].text.split("\n")[0]

    def test_a_merged_header_never_names_only_one_of_its_sections(self):
        doc = _document(DECISION_TEXT, "patent_trial_decision")
        chunks, _ = chunk_imported_document(doc, "ip-v2", OPTS)
        for chunk in chunks:
            header = chunk.text.split("\n")[0]
            if "(1)" in chunk.text and "(2)" in chunk.text:
                assert not header.rstrip().endswith("(1)")

    def test_top_level_sections_are_never_merged_together(self):
        doc = _document("제1조(목적)\n짧다.\n제2조(정의)\n짧다.\n제3조(적용)\n짧다.\n", "statute")
        chunks, _ = chunk_imported_document(doc, "ip-v2", OPTS)
        assert len(chunks) == 3

    def test_oversized_leaf_splits_at_sentence_boundaries(self):
        doc = _document(STATUTE_TEXT, "statute")
        chunks, _ = chunk_imported_document(doc, "ip-v2", OPTS)
        parts = [c for c in chunks if c.text.startswith("상표법 제2조(정의)")]
        assert len(parts) > 1
        for part in parts:
            body = part.text.split("\n", 1)[1]
            assert body.endswith(".") or body.endswith("다")


class TestFallback:
    def test_unstructured_text_falls_back_to_windows(self):
        doc = _document(UNSTRUCTURED, "judgment")
        chunks, report = chunk_imported_document(doc, "ip-v2", OPTS)
        assert not report.parsed and report.reason == "no structure"
        assert len(chunks) > 1
        assert all(c.text.startswith("doc-1 상표법\n") for c in chunks)

    def test_fallback_chunks_still_carry_a_header(self):
        chunks, _ = chunk_imported_document(_document(UNSTRUCTURED, "judgment"), "ip-v2", OPTS)
        assert chunks[0].text.split("\n")[0] == "doc-1 상표법"


class TestIdentityAndMetadata:
    def test_ids_are_unique_and_prefixed_by_the_document_id(self):
        doc = _document(DECISION_TEXT, "patent_trial_decision", id="2011101003196")
        chunks, _ = chunk_imported_document(doc, "ip-v2", OPTS)
        ids = [c.id for c in chunks]
        assert len(set(ids)) == len(ids)
        assert all(i.startswith("2011101003196#") for i in ids)

    def test_ids_are_stable_across_runs(self):
        doc = _document(DECISION_TEXT, "patent_trial_decision")
        first, _ = chunk_imported_document(doc, "ip-v2", OPTS)
        second, _ = chunk_imported_document(doc, "ip-v2", OPTS)
        assert [c.id for c in first] == [c.id for c in second]

    def test_order_is_contiguous_from_zero(self):
        chunks, _ = chunk_imported_document(_document(DECISION_TEXT, "judgment"), "ip-v2", OPTS)
        assert [c.order for c in chunks] == list(range(len(chunks)))

    def test_document_fields_are_preserved(self):
        doc = _document(DECISION_TEXT, "judgment", id="d9", title="상표법위반")
        chunks, _ = chunk_imported_document(doc, "ip-v2", OPTS)
        chunk = chunks[0]
        assert chunk.document_id == "d9"
        assert chunk.document_type is DocumentType.judgment
        assert chunk.title == "상표법위반"
        assert chunk.metadata.legal_domain == "intellectual_property"
        assert chunk.metadata.corpus_version == "ip-v2"
        assert chunk.metadata.dataset == "aihub_preprocessed"

    def test_original_header_is_carried_through(self):
        original = {"authority": "특허심판원", "title": "게시 시스템", "metadata": {"case_number": "2011허1234"}}
        doc = _document(DECISION_TEXT, "patent_trial_decision", original=original)
        chunks, _ = chunk_imported_document(doc, "ip-v2", OPTS)
        assert chunks[0].metadata.original == original
        assert chunks[0].metadata.authority == "특허심판원"
        assert chunks[0].metadata.court == "특허심판원"
        assert chunks[0].metadata.case_number == "2011허1234"

    def test_statute_metadata_records_law_and_article(self):
        chunks, _ = chunk_imported_document(_document(STATUTE_TEXT, "statute"), "ip-v2", OPTS)
        assert chunks[0].metadata.law_name == "상표법"
        assert chunks[0].metadata.article == "제1조"

    def test_section_metadata_holds_the_readable_path(self):
        chunks, _ = chunk_imported_document(_document(DECISION_TEXT, "judgment"), "ip-v2", OPTS)
        assert any(" > " in (c.metadata.section or "") for c in chunks)


class TestCoverage:
    @pytest.mark.parametrize(
        "text,kind", [(STATUTE_TEXT, "statute"), (DECISION_TEXT, "patent_trial_decision"), (UNSTRUCTURED, "judgment")]
    )
    def test_no_source_text_is_dropped(self, text, kind):
        chunks, _ = chunk_imported_document(_document(text, kind), "ip-v2", OPTS)
        letters = re.compile(r"[0-9A-Za-z가-힣]")
        produced = Counter(letters.findall("".join(c.text for c in chunks)))
        assert not (Counter(letters.findall(text)) - produced)


class TestOptions:
    def test_rejects_incoherent_sizes(self):
        with pytest.raises(ValueError):
            ChunkingOptions(target_chars=2000, max_chars=1200, merge_threshold=400)

    def test_smaller_target_produces_more_chunks(self):
        doc = _document(DECISION_TEXT, "patent_trial_decision")
        big, _ = chunk_imported_document(doc, "ip-v2", ChunkingOptions(2400, 3000, 400))
        small, _ = chunk_imported_document(doc, "ip-v2", ChunkingOptions(400, 800, 100))
        assert len(small) > len(big)

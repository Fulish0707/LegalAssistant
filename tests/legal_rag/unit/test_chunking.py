import json
from pathlib import Path

from legal_rag.chunking import chunk_document
from legal_rag.chunking.fallback import window_sentences
from legal_rag.ingestion.normalize import normalize

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "raw"


def _doc(schema, rel, dataset):
    return normalize(schema, json.loads((FIXTURES / rel).read_text(encoding="utf-8")), dataset=dataset, file=rel)


def test_short_article_is_one_chunk_and_long_article_splits_by_paragraph():
    statute = _doc("statute_json_v1", "statutes/housing_lease_act.json", "statutes")
    chunks = chunk_document(statute, corpus_version="v1")
    paths = [c.path for c in chunks]
    assert "art:제3조" in paths
    assert "art:제3조의3/para:제1항" in paths and "art:제3조의3/para:제2항" in paths
    assert not any(p.startswith("art:제3조의3/para:제2항/") for p in paths)
    assert all(c.metadata.law_name == "주택임대차보호법" for c in chunks)
    para2 = next(c for c in chunks if c.path == "art:제3조의3/para:제2항")
    assert "제1호 신청의 취지 및 이유" in para2.text
    assert para2.text.startswith("주택임대차보호법 제3조의3(임차권등기명령) 제2항")
    assert para2.metadata.paragraph == "제2항" and para2.metadata.article == "제3조의3"
    assert [c.order for c in chunks] == list(range(len(chunks)))
    assert all(c.id == f"{statute.id}#{c.path}" for c in chunks)


def test_decision_chunks_by_section_and_windows_long_sections():
    decision = _doc("aihub_judgment_v1", "aihub_judgment/judgment_002.json", "aihub_judgment")
    chunks = chunk_document(decision, corpus_version="v1")
    paths = [c.path for c in chunks]
    assert "sec:facts" in paths and "sec:holding" in paths
    assert "sec:reasoning/1" in paths and "sec:reasoning/2" in paths
    reasoning = [c for c in chunks if c.path.startswith("sec:reasoning")]
    assert all(c.metadata.case_number == "2018다12345" and c.metadata.section == "reasoning" for c in reasoning)
    assert reasoning[0].text.startswith("대법원 2018다12345 (2019-05-30) — 판단")
    assert all(len(c.text) <= 1200 for c in chunks)


def test_admin_interpretation_chunks_question_and_answer_together_when_short():
    doc = _doc("admin_interpretation_v1", "admin_interpretations/interp_001.json", "admin_interpretations")
    chunks = chunk_document(doc, corpus_version="v1")
    assert [c.path for c in chunks] == ["sec:qa"]
    assert chunks[0].metadata.authority == "법제처" and "[회답]" in chunks[0].text


def test_window_sentences_overlaps_by_one_sentence():
    sentences = [f"문장{i}입니다." for i in range(6)]
    windows = window_sentences(sentences, max_chars=20)
    assert windows[0] == "문장0입니다. 문장1입니다."
    assert windows[1].startswith("문장1입니다.")
    assert windows[-1].endswith("문장5입니다.")

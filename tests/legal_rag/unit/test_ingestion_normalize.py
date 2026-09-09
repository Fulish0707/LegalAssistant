import json
from datetime import date
from pathlib import Path

import pytest

from legal_rag.ingestion.normalize import NormalizationError, normalize
from legal_rag.schemas import AdminInterpretation, CourtDecision, DecisionSection, Statute

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "raw"


def _load(rel):
    return json.loads((FIXTURES / rel).read_text(encoding="utf-8"))


def test_judgment_normalizes_sections_and_refs():
    doc = normalize("aihub_judgment_v1", _load("aihub_judgment/judgment_001.json"), dataset="aihub_judgment", file="aihub_judgment/judgment_001.json")
    assert isinstance(doc, CourtDecision)
    assert doc.case_number == "2019가단12345" and doc.court == "서울중앙지방법원"
    assert doc.decision_date == date(2020, 3, 15)
    assert set(doc.sections) == {DecisionSection.facts, DecisionSection.reasoning, DecisionSection.holding, DecisionSection.issues}
    assert "주택임대차보호법 제3조의3" in doc.referenced_statutes
    assert doc.legal_domain == "민사"
    assert doc.text.startswith("서울중앙지방법원 2019가단12345")
    assert "[판단]" in doc.text and doc.source.record_key == "2019가단12345"


def test_statute_normalizes_hierarchy():
    doc = normalize("statute_json_v1", _load("statutes/housing_lease_act.json"), dataset="statutes", file="statutes/housing_lease_act.json")
    assert isinstance(doc, Statute)
    art = {a.number: a for a in doc.articles}["제3조의3"]
    assert art.paragraphs[1].items[0].number == "제1호"
    assert doc.effective_date == date(2020, 12, 10)
    assert doc.source.record_key == "주택임대차보호법@법률 제17363호"
    assert doc.text.startswith("주택임대차보호법\n제3조(대항력 등)")


def test_statute_numbering_accepts_bare_digits():
    payload = {"law_name": "테스트법", "articles": [{"number": "제 1 조", "text": "본문", "paragraphs": [{"number": "1", "text": "항", "items": [{"number": "2", "text": "호"}]}]}]}
    doc = normalize("statute_json_v1", payload, dataset="d", file="f")
    assert doc.articles[0].number == "제1조"
    assert doc.articles[0].paragraphs[0].number == "제1항" and doc.articles[0].paragraphs[0].items[0].number == "제2호"


def test_admin_interpretation_normalizes():
    doc = normalize("admin_interpretation_v1", _load("admin_interpretations/interp_001.json"), dataset="admin_interpretations", file="x.json")
    assert isinstance(doc, AdminInterpretation) and doc.reference_number == "19-0123"
    assert doc.date == date(2019, 8, 1) and "[회답]" in doc.text


def test_missing_case_number_raises():
    payload = _load("aihub_judgment/judgment_001.json")
    payload["info"].pop("caseNo")
    with pytest.raises(NormalizationError):
        normalize("aihub_judgment_v1", payload, dataset="d", file="f")


def test_unknown_schema_raises():
    with pytest.raises(NormalizationError):
        normalize("nope", {}, dataset="d", file="f")

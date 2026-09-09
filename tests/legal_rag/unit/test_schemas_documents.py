from datetime import date

import pytest
from pydantic import TypeAdapter, ValidationError

from legal_rag.schemas import (
    AdminInterpretation,
    CourtDecision,
    DecisionSection,
    DocumentType,
    LegalDocument,
    SourceRef,
    Statute,
    StatuteArticle,
    make_document_id,
)


def _source():
    return SourceRef(dataset="fixture", file="a.json", record_key="k1")


def test_document_id_is_deterministic():
    a = make_document_id(DocumentType.statute, "aihub", "민법@2020")
    b = make_document_id(DocumentType.statute, "aihub", "민법@2020")
    assert a == b and a.startswith("statute:") and len(a) == len("statute:") + 16
    assert make_document_id(DocumentType.statute, "other", "민법@2020") != a


def test_statute_round_trips_through_discriminated_union():
    statute = Statute(
        id=make_document_id(DocumentType.statute, "fixture", "민법"),
        title="민법", text="제750조 ...", source=_source(), law_name="민법",
        articles=[StatuteArticle(number="제750조", title="불법행위의 내용", text="고의 또는 과실로...")],
    )
    parsed = TypeAdapter(LegalDocument).validate_json(statute.model_dump_json())
    assert isinstance(parsed, Statute) and parsed.articles[0].number == "제750조"
    assert parsed.document_type == DocumentType.statute


def test_court_decision_requires_known_sections():
    with pytest.raises(ValidationError):
        CourtDecision(
            id="court_decision:x", title="t", text="t", source=_source(), court="대법원",
            case_number="2018다12345", decision_date=date(2019, 5, 30),
            sections={"bogus": "..."},
        )
    ok = CourtDecision(id="court_decision:x", title="t", text="t", source=_source(), court="대법원",
                       case_number="2018다12345", sections={DecisionSection.holding: "주문"})
    assert ok.sections[DecisionSection.holding] == "주문"


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        AdminInterpretation(
            id="admin_interpretation:x", title="t", text="t", source=_source(), authority="법제처",
            reference_number="19-0001", subject="s", question="q", interpretation="i", surprise=1,
        )

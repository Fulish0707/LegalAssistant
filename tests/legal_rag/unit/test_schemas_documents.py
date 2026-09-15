from datetime import date

import pytest
from pydantic import TypeAdapter, ValidationError

from legal_rag.schemas import (
    AdminInterpretation,
    CitationKind,
    CourtDecision,
    DecisionSection,
    DocumentType,
    LegalDocument,
    SourceRef,
    Statute,
    StatuteArticle,
    citation_kind,
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


def test_citation_kind_covers_every_document_type():
    # Regression guard: a new DocumentType member must be classified, not silently
    # fall through to "interpretation" the way judgment/decision once did.
    for document_type in DocumentType:
        assert isinstance(citation_kind(document_type), CitationKind)


def test_citation_kind_groups_adjudicative_types_as_case():
    case_like = {
        DocumentType.judgment,
        DocumentType.court_decision,
        DocumentType.decision,
        DocumentType.patent_trial_decision,
        DocumentType.constitutional_decision,
    }
    for document_type in case_like:
        assert citation_kind(document_type) is CitationKind.case
    assert citation_kind(DocumentType.statute) is CitationKind.statute
    assert citation_kind(DocumentType.legal_interpretation) is CitationKind.interpretation
    assert citation_kind(DocumentType.admin_interpretation) is CitationKind.interpretation

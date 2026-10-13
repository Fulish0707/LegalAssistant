import pytest

from legal_rag.schemas import Citation, CitationKind, DocumentType, Evidence, EvidenceBundle, LegalAnswer
from legal_rag.textutil.refs import CASE_RE, extract_case_refs
from legal_rag.verification import Verifier, backfill_citations


def bundle(text="대법원 99다 40418 판결", header=None):
    return EvidenceBundle(items=[Evidence(source_id="S1", chunk_id="c", document_id="d",
        document_type=DocumentType.court_decision, header=header or {"Type": "Case"}, text=text, score=1.0)])


def answer(text="대법원 99다40418 판결에 따른 설명입니다.", source="S1", reference="99다40418"):
    return LegalAnswer(analysis=text, summary="제공 근거를 참고하세요.", confidence="low",
        citations=[Citation(source_id=source, kind=CitationKind.case, reference=reference)])


@pytest.mark.parametrize("text", ["99다 40418", "99 다40418", "99 다 40418", "99다\t40418", "99다\u00a040418", "99다\u300040418"])
def test_case_horizontal_spacing_is_canonical(text):
    assert extract_case_refs(text) == ["99다40418"]
    assert extract_case_refs(text + ", 99다40418") == ["99다40418"]


def test_spacing_preserves_case_type_and_numeric_boundaries():
    assert extract_case_refs("2020 헌마 843, 2018 가합 12345, 2020헌마843") == ["2020헌마843", "2018가합12345"]
    for text in ["99다\n40418", "99\n다40418", "12345다12345", "2018다12345678", "2019. 5. 30."]:
        assert extract_case_refs(text) == []


def test_shared_case_regex_keeps_string_findall_contract():
    # Retrieval tokenization and answer evaluation also use the shared regex.
    assert CASE_RE.findall("99다 40418, 2020헌마843") == ["2020헌마843"]


def test_evidence_spacing_does_not_trigger_regeneration():
    report = Verifier().verify(answer(), bundle())
    assert report.ok and report.checked_cases == ["99다40418"]


def test_spaced_header_and_answer_are_compared_canonically():
    assert Verifier().verify(answer(reference="99 다 40418"), bundle("관련 설명", {"Case number": "99 다 40418"})).ok


@pytest.mark.parametrize("reference", ["99다 40419", "99도 40418", "2000다 40418"])
def test_different_spaced_cases_are_still_rejected(reference):
    report = Verifier().verify(answer(reference=reference, text=reference), bundle())
    assert {i.kind for i in report.issues} == {"source_id_mismatch", "case_not_in_evidence"}


def test_spaced_reference_cannot_use_a_different_source():
    b = bundle()
    other = b.items[0].model_copy(update={"source_id": "S2", "text": "2019가단12345"})
    report = Verifier().verify(answer(source="S2", reference="99다 40418"), EvidenceBundle(items=[*b.items, other]))
    assert [i.kind for i in report.issues] == ["source_id_mismatch"]
    assert {i.kind for i in Verifier().verify(answer(source="S9"), b).issues} == {"unknown_source_id"}


def test_prior_and_user_case_spacing_excuse_only_the_same_number():
    empty = EvidenceBundle()
    a = answer().model_copy(update={"citations": []})
    assert Verifier().verify(a, empty, prior_refs=["99다 40418"]).ok
    assert Verifier().verify(a, empty, user_refs=["99 다 40418"]).ok
    assert not Verifier().verify(a, empty, prior_refs=["99다 40419"]).ok


def test_backfill_uses_spaced_header_and_avoids_duplicate_case_labels():
    b = bundle("관련 설명", {"Case number": "99 다 40418"})
    a = answer().model_copy(update={"citations": []})
    filled = backfill_citations(a, b)
    assert [(c.source_id, c.reference) for c in filled.citations] == [("S1", "99다40418")]
    assert backfill_citations(filled, b) == filled
    existing = answer(reference="대법원 99다 40418 판결")
    assert backfill_citations(existing, b) == existing

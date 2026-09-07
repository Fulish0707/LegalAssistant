from legal_rag.schemas import Citation, CitationKind, Confidence, DocumentType, Evidence, EvidenceBundle, LegalAnswer, VerificationAction
from legal_rag.verification import VerificationPolicy, Verifier


def _bundle():
    return EvidenceBundle(items=[
        Evidence(source_id="S1", chunk_id="c1", document_id="d1", document_type=DocumentType.statute,
                 header={"Type": "Statute", "Law": "주택임대차보호법", "Article": "제3조의3"},
                 text="임대차가 끝난 후 보증금이 반환되지 아니한 경우 임차인은 임차권등기명령을 신청할 수 있다.", score=1.0),
        Evidence(source_id="S2", chunk_id="c2", document_id="d2", document_type=DocumentType.court_decision,
                 header={"Type": "Case", "Court": "서울중앙지방법원", "Case number": "2019가단12345"},
                 text="민법 제750조에 따른 손해배상 책임이 인정된다.", score=0.9),
    ])


def _answer(**kw):
    base = dict(
        summary="보증금을 돌려받을 수 있습니다.",
        legal_issues=["보증금 반환"],
        analysis="주택임대차보호법 제3조의3에 따라 임차권등기명령을 신청할 수 있습니다.",
        possible_actions=["임차권등기명령 신청"],
        additional_information_needed=[],
        citations=[Citation(source_id="S1", kind=CitationKind.statute, reference="주택임대차보호법 제3조의3")],
        uncertainty="",
        confidence=Confidence.high,
    )
    base.update(kw)
    return LegalAnswer(**base)


def test_grounded_answer_passes():
    report = Verifier().verify(_answer(), _bundle())
    assert report.ok and report.issues == [] and report.checked_statutes == ["주택임대차보호법 제3조의3"]


def test_unknown_source_id_and_foreign_statute_and_case_are_flagged():
    ans = _answer(
        citations=[Citation(source_id="S9", kind=CitationKind.statute, reference="민법 제750조")],
        analysis="민법 제839조의2와 대법원 2015다99999 판결에 따르면 재산분할을 청구할 수 있습니다.",
    )
    report = Verifier().verify(ans, _bundle())
    assert {i.kind for i in report.issues} == {"unknown_source_id", "statute_not_in_evidence", "case_not_in_evidence"}
    assert "민법 제839조의2" in " ".join(i.detail for i in report.issues)
    assert report.checked_cases == ["2015다99999"]


def test_source_id_mismatch_when_reference_not_in_cited_block():
    ans = _answer(citations=[Citation(source_id="S1", kind=CitationKind.statute, reference="민법 제750조")], analysis="민법 제750조 관련 설명")
    report = Verifier().verify(ans, _bundle())
    assert [i.kind for i in report.issues] == ["source_id_mismatch"]


def test_statute_mentioned_inside_evidence_text_counts_as_supported():
    ans = _answer(analysis="민법 제750조에 따라 손해배상 책임이 있습니다.",
                  citations=[Citation(source_id="S2", kind=CitationKind.statute, reference="민법 제750조")])
    assert Verifier().verify(ans, _bundle()).ok


def test_case_citation_supported_by_header():
    ans = _answer(analysis="서울중앙지방법원 2019가단12345 판결도 같은 취지입니다.",
                  citations=[Citation(source_id="S2", kind=CitationKind.case, reference="2019가단12345")])
    assert Verifier().verify(ans, _bundle()).ok


def test_policy_regenerates_then_strips_then_rejects():
    ans = _answer(
        citations=[Citation(source_id="S9", kind=CitationKind.statute, reference="민법 제839조의2")],
        analysis="민법 제839조의2에 따라 재산분할을 청구할 수 있습니다. 또한 보증금은 반환되어야 합니다.",
        possible_actions=["제839조의2에 따른 청구", "보증금 반환 청구"],
    )
    report = Verifier().verify(ans, _bundle())
    policy = VerificationPolicy(max_regenerations=1)
    assert policy.decide(report, attempt=0) == VerificationAction.regenerate
    assert policy.decide(report, attempt=1) == VerificationAction.strip
    assert "민법 제839조의2" in policy.feedback(report)
    stripped = policy.strip(ans, report)
    assert stripped.citations == []
    assert "제839조의2" not in stripped.analysis and "보증금은 반환되어야 합니다." in stripped.analysis
    assert stripped.possible_actions == ["보증금 반환 청구"]
    assert stripped.confidence == Confidence.low and "제839조의2" in stripped.uncertainty
    assert policy.decide_after_strip(stripped, evidence_required=True) == VerificationAction.strip
    empty = stripped.model_copy(update={"analysis": ""})
    assert policy.decide_after_strip(empty, evidence_required=True) == VerificationAction.reject


def test_accept_when_ok():
    assert VerificationPolicy().decide(Verifier().verify(_answer(), _bundle()), attempt=0) == VerificationAction.accept

from legal_rag.schemas import Citation, CitationKind, Confidence, DocumentType, Evidence, EvidenceBundle, LegalAnswer, VerificationAction
from legal_rag.verification import VerificationPolicy, Verifier, backfill_citations
from legal_rag.verification.backfill import _kind
from legal_rag.verification.policy import STRIP_NOTE


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


def test_user_mentioned_reference_is_exempt_from_not_in_evidence_but_not_from_source_id_checks():
    ans = _answer(citations=[], analysis="민법 제839조의2는 제공된 근거에서 확인되지 않습니다. 2015다99999 판결도 마찬가지입니다.")
    assert Verifier().verify(ans, _bundle(), user_refs=["민법 제839조의2", "2015다99999"]).ok
    assert not Verifier().verify(ans, _bundle()).ok
    cited = _answer(citations=[Citation(source_id="S1", kind=CitationKind.statute, reference="민법 제839조의2")], analysis="민법 제839조의2 관련")
    report = Verifier().verify(cited, _bundle(), user_refs=["민법 제839조의2"])
    assert [i.kind for i in report.issues] == ["source_id_mismatch"]


def test_backfill_adds_citations_for_grounded_references_only():
    ans = _answer(citations=[], analysis="주택임대차보호법 제3조의3 제1항과 민법 제750조에 따르면 그렇습니다. 민법 제839조의2는 확인되지 않습니다. 2019가단12345 판결 참조.")
    filled = backfill_citations(ans, _bundle())
    assert [(c.source_id, c.reference, c.kind) for c in filled.citations] == [
        ("S1", "주택임대차보호법 제3조의3", CitationKind.statute),
        ("S2", "민법 제750조", CitationKind.case),
        ("S2", "2019가단12345", CitationKind.case),
    ]
    assert backfill_citations(filled, _bundle()) == filled
    assert backfill_citations(_answer(), _bundle()).citations == _answer().citations


def test_backfilled_citation_kind_follows_document_type():
    for document_type in (DocumentType.judgment, DocumentType.patent_trial_decision,
                          DocumentType.constitutional_decision, DocumentType.decision):
        evidence = Evidence(source_id="S1", chunk_id="c", document_id="d",
                            document_type=document_type, header={"Type": "Case"},
                            text="대법원 2018다12345 판결", score=1.0)
        assert _kind(evidence) is CitationKind.case


def test_statutes_in_issue_and_follow_up_lists_are_verified():
    """answer_text skipped legal_issues and additional_information_needed, so "헌법 제11조"
    in a follow-up list passed verification (2026-09-27, 80-question run)."""
    for field in ("legal_issues", "additional_information_needed"):
        report = Verifier().verify(_answer(**{field: ["헌법 제11조 위반 여부"]}), _bundle())
        assert not report.ok and any("제11조" in issue.detail for issue in report.issues), field


def test_strip_cleans_issue_and_follow_up_lists():
    ans = _answer(legal_issues=["보증금 반환", "의료법 제12조 설명의무"],
                  additional_information_needed=["헌법 제11조 적용 여부", "계약 만료일"])
    report = Verifier().verify(ans, _bundle())
    stripped = VerificationPolicy().strip(ans, report)
    assert stripped.legal_issues == ["보증금 반환"]
    assert stripped.additional_information_needed == ["계약 만료일"]


def test_strip_cleans_the_model_written_uncertainty():
    """strip left "석면피해구제법 제3조 제3항의 …" in the model's own uncertainty text, so the
    removed citation still reached the user (2026-09-28, 80-question rerun)."""
    ans = _answer(analysis="보증금은 반환되어야 합니다.",
                  uncertainty="헌법 제11조의 적용 범위는 불확실합니다. 계약 만료일이 필요합니다.")
    report = Verifier().verify(ans, _bundle())
    stripped = VerificationPolicy().strip(ans, report)
    model_text, note = stripped.uncertainty.split(STRIP_NOTE)
    assert "제11조" not in model_text and "계약 만료일이 필요합니다." in model_text
    assert "헌법 제11조" in note


def test_descriptive_label_without_article_or_case_number_is_accepted_when_the_source_exists():
    # Gemini names a block that has no case number by its court ("서울고등법원 판결"); 28 of 29 flagged
    # first drafts in the 2026-10-05 serving check were exactly this.
    for label in ("서울고등법원 판결", "대법원 판결 (사건번호 미기재)", "LEGAL SOURCE S2"):
        ans = _answer(citations=[Citation(source_id="S2", kind=CitationKind.case, reference=label)])
        assert Verifier().verify(ans, _bundle()).ok, label


def test_descriptive_label_still_needs_a_source_that_exists():
    ans = _answer(citations=[Citation(source_id="S9", kind=CitationKind.case, reference="서울고등법원 판결")])
    assert [i.kind for i in Verifier().verify(ans, _bundle()).issues] == ["unknown_source_id"]


def test_a_case_number_in_the_label_is_still_checked_against_the_block():
    ans = _answer(citations=[Citation(source_id="S2", kind=CitationKind.case, reference="대법원 2015다99999 판결")],
                  analysis="대법원 2015다99999 판결 참조")
    kinds = {i.kind for i in Verifier().verify(ans, _bundle()).issues}
    assert "source_id_mismatch" in kinds and "case_not_in_evidence" in kinds


def test_a_reference_from_an_earlier_answer_may_be_repeated_and_excuses_nothing_else():
    """A follow-up repeats 근로기준법 제36조 from the last answer; that must not let 민법 제36조 through."""
    def ok(text):
        return Verifier().verify(_answer(citations=[], analysis=text), _bundle(), prior_refs=["근로기준법 제36조", "2020다11111"]).ok

    assert ok("근로기준법 제36조에 따라 14일 이내에 지급해야 합니다. 2020다11111 판결도 같습니다.")
    assert ok("근로기준법 제36조가 적용됩니다. 제36조의 기한은 14일입니다.")
    assert not ok("민법 제36조에 따라 청구할 수 있습니다.")
    assert not ok("제36조에 따라 청구할 수 있습니다.")
    assert not ok("근로기준법 제36조가 적용됩니다. 제37조의 지연이자도 붙습니다.")
    assert not ok("2021다22222 판결도 같습니다.")

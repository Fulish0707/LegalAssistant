"""Validate model assessments against exact supplied facts and legal passages."""

import importlib

import pytest

from legal_rag.llm import FakeChatModel
from legal_rag.schemas import Evidence, EvidenceBundle, UserDocument


def module():
    try:
        return importlib.import_module("legal_rag.rag.sufficiency")
    except ModuleNotFoundError as error:
        if error.name != "legal_rag.rag.sufficiency":
            raise
        pytest.fail("Sufficiency assessment is not implemented")


def test_document_fact_is_seen_and_does_not_require_reasking():
    s = module()
    result = s.SufficiencyAssessment(
        facts=[s.FactFinding(key="termination_agreement", status="known", value="종료에 합의함",
                             reason="첨부에 명시됨", observations=[s.FactObservation(locator="D1", quote="종료에 합의함")])],
        evidence=[s.EvidenceFinding(key="deduction_basis", proposition="합의 종료 뒤 차임 공제의 근거",
                                   status="insufficient", reason="검색 결과 없음")],
    )
    model = FakeChatModel({"SufficiencyAssessment": [result]})
    assessor = s.SufficiencyAssessor(model)
    outcome = assessor.assess("공제 가능한가요?", [], [UserDocument(name="합의서", text="종료에 합의함")],
                              EvidenceBundle(), issue_id="rent-deduction", matter_id="lease-1")
    assert outcome.policy_input.facts[0].status == "known"
    assert outcome.policy_input.evidence[0].status == "insufficient"
    prompt = "\n".join(m.content for m in model.calls[0].messages)
    assert "종료에 합의함" in prompt
    assert "D1" in prompt


@pytest.mark.parametrize("problem", ["invented_quote", "wrong_locator", "missing_fact", "wrong_fact_key", "missing_law", "duplicate_law", "empty_fact_key"])
def test_unverifiable_or_incomplete_assessment_cannot_authorize_answer(problem):
    s = module()
    fact = s.FactFinding(key="termination_agreement", status="known", value="종료 합의 있음",
                         reason="문서", observations=[s.FactObservation(locator="D1", quote="합의함")])
    law = s.EvidenceFinding(key="deduction_basis", proposition="공제 근거", status="insufficient", reason="자료 없음")
    facts, evidence = [fact], [law]
    if problem == "invented_quote":
        facts = [fact.model_copy(update={"observations": [s.FactObservation(locator="D1", quote="없는 문장")]})]
    elif problem == "wrong_locator":
        facts = [fact.model_copy(update={"observations": [s.FactObservation(locator="S1", quote="합의함")]})]
    elif problem == "missing_fact":
        facts = []
    elif problem == "wrong_fact_key":
        facts = [fact.model_copy(update={"key": "deposit_amount"})]
    elif problem == "missing_law":
        evidence = []
    elif problem == "empty_fact_key":
        facts = [fact.model_copy(update={"key": ""})]
    else:
        # Same key, contradictory content: genuinely ambiguous, must fail closed.
        evidence = [law, law.model_copy(update={"status": "conflicting"})]
    model = FakeChatModel({"SufficiencyAssessment": [s.SufficiencyAssessment(facts=facts, evidence=evidence)]})
    outcome = s.SufficiencyAssessor(model).assess(
        "공제 가능한가요?", [], [UserDocument(name="합의서", text="합의함")], EvidenceBundle(),
        issue_id="rent-deduction", matter_id="lease-1",
    )
    assert outcome.policy_input.assessment_status == "invalid"


@pytest.mark.parametrize("problem", ["none", "no_support", "wrong_source", "wrong_quote"])
def test_supported_law_requires_a_matching_original_passage(problem):
    s = module()
    support = s.LegalSupport(source_id="S1", quote="검토할 원문")
    supports = [support]
    if problem == "no_support":
        supports = []
    elif problem == "wrong_source":
        supports = [support.model_copy(update={"source_id": "D1"})]
    elif problem == "wrong_quote":
        supports = [support.model_copy(update={"quote": "원문에 없는 지원"})]
    assessment = s.SufficiencyAssessment(facts=[], evidence=[s.EvidenceFinding(
        key="clause_meaning", proposition="특약 의미", status="supported", reason="원문 지원", supports=supports,
    )])
    model = FakeChatModel({"SufficiencyAssessment": [assessment]})
    bundle = EvidenceBundle(items=[Evidence(source_id="S1", chunk_id="c1", document_id="law1",
                                           document_type="statute", header={}, text="검토할 원문", score=0.1)])
    outcome = s.SufficiencyAssessor(model).assess("특약의 의미?", [], [], bundle,
                                                issue_id="clause-explanation", matter_id="lease-1")
    assert outcome.policy_input.assessment_status == ("valid" if problem == "none" else "invalid")


def test_malformed_model_output_is_a_failed_assessment_not_a_fact_gap():
    s = module()
    model = FakeChatModel({"SufficiencyAssessment": ["not json"]})
    outcome = s.SufficiencyAssessor(model).assess("특약의 의미?", [], [], EvidenceBundle(),
                                                issue_id="clause-explanation", matter_id="lease-1")
    assert outcome.policy_input.assessment_status == "invalid"




def test_assessor_sees_the_same_case_figure_mask_as_generator():
    s = module()
    assessment = s.SufficiencyAssessment(facts=[], evidence=[s.EvidenceFinding(
        key="clause_meaning", proposition="특약 의미", status="insufficient", reason="지원 없음",
    )])
    model = FakeChatModel({"SufficiencyAssessment": [assessment]})
    bundle = EvidenceBundle(items=[Evidence(source_id="S1", chunk_id="case1", document_id="case",
        document_type="court_decision", header={"Type": "Case"},
        text="2026년 1월 1일 500만 원의 지급을 명했다.", score=1.0)])
    s.SufficiencyAssessor(model).assess("특약 의미?", [], [], bundle,
        issue_id="clause-explanation", matter_id="lease-1", mask_case_figures=True)
    prompt = "\n".join(m.content for m in model.calls[0].messages)
    assert "500만 원" not in prompt
    assert "2026년 1월 1일" not in prompt
    assert "(금액)" in prompt


def test_identical_duplicate_findings_are_collapsed_not_failed():
    s = module()
    finding = s.EvidenceFinding(key="clause_meaning", proposition="특약 의미", status="insufficient", reason="지원 없음")
    model = FakeChatModel({"SufficiencyAssessment": [s.SufficiencyAssessment(facts=[], evidence=[finding, finding])]})
    outcome = s.SufficiencyAssessor(model).assess("특약 의미?", [], [], EvidenceBundle(),
        issue_id="clause-explanation", matter_id="lease-1")
    assert outcome.policy_input.assessment_status == "valid"
    assert [e.key for e in outcome.policy_input.evidence] == ["clause_meaning"]


def test_user_statement_of_no_agreement_is_a_known_fact():
    s = module()
    assessment = s.SufficiencyAssessment(
        facts=[s.FactFinding(key="termination_agreement", status="known", value="별도 종료 합의 없음",
                             reason="사용자 진술", observations=[s.FactObservation(locator="M0", quote="합의는 없었습니다")])],
        evidence=[s.EvidenceFinding(key="deduction_basis", proposition="공제 근거", status="insufficient", reason="없음")])
    model = FakeChatModel({"SufficiencyAssessment": [assessment]})
    outcome = s.SufficiencyAssessor(model).assess("별도의 조기 종료 합의는 없었습니다. 공제되나요?", [], [],
        EvidenceBundle(), issue_id="rent-deduction", matter_id="lease-1")
    assert outcome.policy_input.facts[0].status == "known"
    prompt = model.calls[0].messages[0].content
    assert "직접 진술" in prompt and "known" in prompt


@pytest.mark.parametrize("name", ["FactFinding", "EvidenceFinding"])
def test_model_writes_its_reasoning_and_sources_before_the_verdict(name):
    """Structured output is generated in schema order. With status first, qwen3:8b fixed
    `missing` and then reasoned its way to `known` in prose the policy never reads."""
    s = module()
    order = list(getattr(s, name).model_json_schema()["properties"])
    support = "observations" if name == "FactFinding" else "supports"
    assert order.index("reason") < order.index("status")
    assert order.index(support) < order.index("status")


@pytest.mark.parametrize("name,field", [("FactFinding", "observations"), ("EvidenceFinding", "supports")])
def test_grammar_forces_the_model_to_emit_its_evidence_field(name, field):
    """With a default, qwen3:8b and EXAONE dropped `observations` and every known fact failed."""
    s = module()
    assert field in getattr(s, name).model_json_schema()["required"]


def test_model_never_copies_chunk_ids():
    """qwen3:8b rewrote 2026-03-17 as 2026-03,17. The server already knows S1's chunk."""
    s = module()
    assert "chunk_id" not in s.LegalSupport.model_json_schema()["properties"]


def test_fact_value_is_a_short_label_enforced_by_the_grammar():
    """EXAONE averaged 209 characters in `value`; Ollama enforces maxLength while decoding."""
    s = module()
    assert s.FactFinding.model_json_schema()["properties"]["value"]["maxLength"] <= 40


def test_prompt_gives_no_quotable_example_of_a_user_statement():
    """qwen3:8b and EXAONE both reported the prompt's example sentence as something the user said."""
    s = module()
    model = FakeChatModel({"SufficiencyAssessment": ["not json"]})
    s.SufficiencyAssessor(model).assess("공제되나요?", [], [], EvidenceBundle(),
        issue_id="rent-deduction", matter_id="lease-1")
    prompt = model.calls[0].messages[0].content
    for copied in ("'합의가 없었다'", "'별도 종료 합의 없음'"):
        assert copied not in prompt


def _case_bundle(role="기관·법원 판단", text="2026년 1월 1일 500만 원의 지급을 명했다."):
    return EvidenceBundle(items=[Evidence(source_id="S1", chunk_id="case1", document_id="case",
        document_type="court_decision", header={"Type": "Case", "Section role": role}, text=text, score=1.0)])


def _assess_clause(s, quote, bundle, mask=True):
    assessment = s.SufficiencyAssessment(facts=[], evidence=[s.EvidenceFinding(
        key="clause_meaning", proposition="특약 의미", status="supported", reason="판례 판단",
        supports=[s.LegalSupport(source_id="S1", quote=quote)])])
    model = FakeChatModel({"SufficiencyAssessment": [assessment]})
    return s.SufficiencyAssessor(model).assess("특약 의미?", [], [], bundle,
        issue_id="clause-explanation", matter_id="lease-1", mask_case_figures=mask)


@pytest.mark.parametrize("quote,expected", [
    ("(날짜) (금액)의 지급을 명했다", "valid"),     # exactly what the model was shown
    ("2026년 1월 1일 500만 원의 지급", "invalid"),  # figures the model never saw
])
def test_case_quotes_are_checked_against_the_masked_text_the_model_read(quote, expected):
    s = module()
    assert _assess_clause(s, quote, _case_bundle()).policy_input.assessment_status == expected


def test_unmasked_runs_still_check_the_original_text():
    s = module()
    assert _assess_clause(s, "2026년 1월 1일 500만 원의 지급", _case_bundle(), mask=False).policy_input.assessment_status == "valid"


@pytest.mark.parametrize("role,expected", [
    ("기관·법원 판단", "valid"), ("법령·관련 법리", "valid"),
    ("당사자 주장 (채택 여부 미확인)", "invalid"), ("사실관계", "invalid"), ("주문·결론", "invalid"),
    ("복수 역할 (본문 구분 확인 필요)", "invalid"),
    # Every operational case chunk is unlabelled; refusing unknown would switch case law off.
    ("역할 미상 (판단·주장 구분 미확인)", "valid"),
])
def test_case_support_blocks_known_non_reasoning_roles_and_allows_unknown(role, expected):
    """A party's claim or the facts quoted verbatim are still not the court's reasoning."""
    s = module()
    bundle = _case_bundle(role=role, text="임차인은 차임을 지급할 의무가 있다.")
    assert _assess_clause(s, "차임을 지급할 의무", bundle).policy_input.assessment_status == expected


def test_law_quote_is_a_short_anchor_enforced_by_the_grammar():
    """Two models mis-copied full articles and appended commentary inside the quote.
    Grammar truncation keeps a correctly copied prefix a valid substring."""
    s = module()
    assert s.LegalSupport.model_json_schema()["properties"]["quote"]["maxLength"] <= 80


class _Documents:
    """Stands in for the vector stores: chunk id -> the whole document text."""
    def __init__(self, text):
        self.text, self.calls = text, []

    def __call__(self, chunk_id):
        self.calls.append(chunk_id)
        return self.text


def _windowed_case(document, start, end):
    # An operational chunk: a fixed window cut mid-document, header role always unknown.
    return EvidenceBundle(items=[Evidence(source_id="S1", chunk_id="case1", document_id="case",
        document_type="court_decision", header={"Type": "Case"}, text=document[start:end], score=1.0)])


def _assess_with_documents(s, quote, bundle, documents, mask=False):
    assessment = s.SufficiencyAssessment(facts=[], evidence=[s.EvidenceFinding(
        key="clause_meaning", proposition="특약 의미", status="supported", reason="판례",
        supports=[s.LegalSupport(source_id="S1", quote=quote)])])
    model = FakeChatModel({"SufficiencyAssessment": [assessment]})
    return s.SufficiencyAssessor(model, document_text=documents).assess("특약 의미?", [], [], bundle,
        issue_id="clause-explanation", matter_id="lease-1", mask_case_figures=mask)


_DOC = ("1. 기초사실\n원고는 2020. 3. 1. 피고와 임대차계약을 체결하였다.\n"
        "2. 당사자의 주장\n원고는 중도 퇴거 후에는 차임을 지급할 의무가 없다고 주장한다.\n"
        "3. 판단\n임차인은 계약이 종료될 때까지 차임을 지급할 의무가 있다.\n")


def test_a_party_claim_found_through_the_whole_document_is_not_support():
    """The window starts mid-section, so the heading that names the speaker is in an earlier chunk."""
    s = module()
    at = _DOC.index("원고는 중도")
    result = _assess_with_documents(s, "차임을 지급할 의무가 없다", _windowed_case(_DOC, at, at + 40), _Documents(_DOC))
    assert result.policy_input.assessment_status == "invalid"
    assert result.support_roles == {"S1": "party_claim"}


def test_the_courts_reasoning_found_through_the_whole_document_is_support():
    s = module()
    at = _DOC.index("임차인은")
    result = _assess_with_documents(s, "차임을 지급할 의무가 있다", _windowed_case(_DOC, at, len(_DOC)), _Documents(_DOC))
    assert result.policy_input.assessment_status == "valid"
    assert result.support_roles == {"S1": "finding"}


def test_quote_is_located_in_the_masked_document_the_model_read():
    s = module()
    at = _DOC.index("원고는 2020")
    bundle = _windowed_case(_DOC, at, _DOC.index("2. 당사자"))
    result = _assess_with_documents(s, "원고는 (날짜) 피고와", bundle, _Documents(_DOC), mask=True)
    assert result.support_roles == {"S1": "facts"}  # found despite the masked date; facts are not support
    assert result.policy_input.assessment_status == "invalid"


def test_without_the_document_the_chunk_alone_decides_and_unknown_is_recorded():
    s = module()
    at = _DOC.index("임차인은")
    for documents in (None, _Documents(None)):
        result = _assess_with_documents(s, "차임을 지급할 의무가 있다", _windowed_case(_DOC, at, len(_DOC)), documents)
        assert result.policy_input.assessment_status == "valid"
        assert result.support_roles == {"S1": "unknown"}


def test_statutes_are_never_looked_up_or_role_checked():
    s = module()
    bundle = EvidenceBundle(items=[Evidence(source_id="S1", chunk_id="law1", document_id="law",
        document_type="statute", header={"Type": "Statute"}, text="임차인은 차임을 지급하여야 한다.", score=1.0)])
    documents = _Documents("원고는 임차인은 차임을 지급하여야 한다고 주장한다.")
    result = _assess_with_documents(s, "차임을 지급하여야 한다", bundle, documents)
    assert result.policy_input.assessment_status == "valid"
    assert documents.calls == [] and result.support_roles == {}


def test_interpretations_get_the_same_claim_filter_as_cases():
    s = module()
    doc = "[질의요지]\n임차인이 중도 퇴거하면 차임 지급 의무가 없다고 볼 수 있는지?\n[회답]\n차임 지급 의무가 있습니다.\n"
    at = doc.index("임차인이")
    bundle = EvidenceBundle(items=[Evidence(source_id="S1", chunk_id="i1", document_id="i",
        document_type="legal_interpretation", header={"Type": "Interpretation"}, text=doc[at:at + 30], score=1.0)])
    result = _assess_with_documents(s, "차임 지급 의무가 없다", bundle, _Documents(doc))
    assert result.support_roles == {"S1": "party_claim"}


_LONG = "임대차는 당사자 일방이 상대방에게 목적물을 사용, 수익하게 할 것을 약정하고 상대방이 이에 대하여 차임을 지급할 것을 약정함으로써 그 효력이 생긴다."


def _law_bundle():
    return EvidenceBundle(items=[Evidence(source_id="S1", chunk_id="law1", document_id="law",
        document_type="statute", header={"Type": "Statute"}, text=_LONG, score=1.0)])


@pytest.mark.parametrize("quote,expected", [
    (_LONG[:79] + "X", "valid"),        # at the limit: the grammar bent only the last character
    (_LONG[:76] + "XXXX", "valid"),     # at most four bent characters
    (_LONG[:75] + "XXXXX", "invalid"),  # more than the forced boundary explains
    (_LONG[:39] + "X", "invalid"),      # below the limit a copy must be exact
    (_LONG[:20] + "X" * 60, "invalid"), # a short true prefix is not an anchor
])
def test_a_quote_cut_by_the_length_limit_may_differ_only_at_its_last_characters(quote, expected):
    """Live runs: six 80-character quotes were exact but for the final character ("것으로보" for "것으로 보아"),
    because near maxLength decoding can only pick tokens that still fit."""
    s = module()
    assert len(quote) <= 80
    assert _assess_with_documents(s, quote, _law_bundle(), None).policy_input.assessment_status == expected


def test_role_of_a_bent_quote_is_read_at_its_exact_prefix():
    s = module()
    doc = _DOC + ("임차인이 계약기간 중에 중도 퇴거하더라도 계약이 당사자의 합의로 종료되지 않은 이상, "
                  "그 이후의 기간에 대한 차임을 지급할 의무는 계속 남는다고 보아야 한다.")
    tail = doc[doc.index("임차인이 계약기간"):]
    assert len(tail) > 80
    quote = tail[:79] + "X"
    result = _assess_with_documents(s, quote, _windowed_case(doc, doc.index("임차인이 계약기간"), len(doc)), _Documents(doc))
    assert result.policy_input.assessment_status == "valid"
    assert result.support_roles == {"S1": "finding"}

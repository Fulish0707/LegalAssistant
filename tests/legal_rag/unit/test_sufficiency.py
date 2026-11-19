"""Structured sufficiency assessment: per-issue fact questions and legal elements.

The model fills a schema built for each request; code derives known/missing and
supported/insufficient from it and checks every quote against what the model was shown.
"""

import importlib
import json

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


# --- scripted model output -------------------------------------------------------------

def said(answer, *quotes, reason="사용자 자료"):
    out = {"reason": reason, "answer": answer}
    if answer != "not_stated":
        out["observations"] = list(quotes)
    return out


def addressed(*supports, reason="자료가 직접 다룸"):
    return {"reason": reason, "verdict": "addressed",
            "supports": [{"source_id": sid, "quote": quote} for sid, quote in supports]}


def not_addressed(reason="자료가 다루지 않음"):
    return {"reason": reason, "verdict": "not_addressed"}


def scripted(facts=None, elements=None):
    return json.dumps({"facts": facts or {}, "elements": elements or {}}, ensure_ascii=False)


def law(text="임차인은 차임을 지급하여야 한다.", sid="S1"):
    return Evidence(source_id=sid, chunk_id=f"law-{sid}", document_id="law", document_type="statute",
                    header={"Type": "Statute"}, text=text, score=1.0)


def assess(output, *, message="공제되나요?", documents=(), bundle=None, issue="rent-deduction",
           document_text=None, mask=False):
    s = module()
    model = FakeChatModel({"StructuredAssessment": [output]})
    result = s.SufficiencyAssessor(model, document_text=document_text).assess(
        message, [], list(documents), bundle if bundle is not None else EvidenceBundle(),
        issue_id=issue, matter_id="lease-1", mask_case_figures=mask)
    return result, model


def schema_of(model):
    return model.calls[0].response_schema.model_json_schema()


def resolve(schema, node):
    while "$ref" in node:
        node = schema["$defs"][node["$ref"].rsplit("/", 1)[1]]
    return node


def branches(schema, node):
    node = resolve(schema, node)
    return [resolve(schema, b) for b in node.get("anyOf", [node])]


def slot(schema, group, key):
    return resolve(schema, schema["properties"][group])["properties"][key]


RENT_ELEMENTS = ("rent_after_end", "clause_scope")


def rent_elements(**override):
    out = {key: not_addressed() for key in RENT_ELEMENTS}
    out.update(override)
    return out


# --- the schema the model fills ---------------------------------------------------------

def test_keys_are_fixed_by_the_issue_so_none_can_be_invented_or_repeated():
    _, model = assess(scripted())
    schema = schema_of(model)
    facts = resolve(schema, schema["properties"]["facts"])
    elements = resolve(schema, schema["properties"]["elements"])
    assert facts["required"] == ["termination_agreement"] and facts["additionalProperties"] is False
    assert elements["required"] == list(RENT_ELEMENTS) and elements["additionalProperties"] is False


def test_clause_explanation_asks_no_fact_question():
    _, model = assess(scripted(), issue="clause-explanation")
    schema = schema_of(model)
    assert resolve(schema, schema["properties"]["facts"]).get("properties", {}) == {}
    assert resolve(schema, schema["properties"]["elements"])["required"] == ["agreement_effect"]


def test_a_legal_support_can_only_name_a_legal_source():
    bundle = EvidenceBundle(items=[law(sid="S1"), law(sid="S2")])
    _, model = assess(scripted(), bundle=bundle, documents=[UserDocument(name="계약서", text="특약")])
    schema = schema_of(model)
    yes = next(b for b in branches(schema, slot(schema, "elements", "rent_after_end"))
               if b["properties"]["verdict"].get("const") == "addressed")
    support = resolve(schema, yes["properties"]["supports"]["items"])
    assert support["properties"]["source_id"]["enum"] == ["S1", "S2"]
    assert "chunk_id" not in support["properties"]
    assert support["properties"]["quote"]["maxLength"] <= 80


def test_the_model_copies_the_users_sentence_and_code_finds_where_it_is():
    """Smoke run: EXAONE quoted the message (M0) exactly but labelled it D1, twice. Like chunk ids,
    the locator is something the server already knows."""
    _, model = assess(scripted(), documents=[UserDocument(name="계약서", text="특약")])
    schema = schema_of(model)
    yes = next(b for b in branches(schema, slot(schema, "facts", "termination_agreement"))
               if b["properties"]["answer"].get("const") == "yes")
    assert resolve(schema, yes["properties"]["observations"]["items"])["type"] == "string"


def test_an_observation_is_recorded_where_it_appears_in_the_users_material():
    output = scripted({"termination_agreement": said("no", "합의는 없었습니다")}, rent_elements())
    result, _ = assess(output, message="합의는 없었습니다. 공제되나요?",
                       documents=[UserDocument(name="계약서", text="특약: 조기 퇴거 시 차임 부담")])
    assert [(o.locator, o.quote) for o in result.assessment.facts[0].observations] == [("M0", "합의는 없었습니다")]


def test_without_legal_sources_an_element_can_only_be_not_addressed():
    _, model = assess(scripted())
    schema = schema_of(model)
    options = branches(schema, slot(schema, "elements", "rent_after_end"))
    assert [b["properties"]["verdict"]["const"] for b in options] == ["not_addressed"]


def test_every_branch_starts_with_the_reason_so_the_verdict_comes_after_it():
    """Spike 001: branches that opened with different fields fixed the verdict at the first key."""
    _, model = assess(scripted(), bundle=EvidenceBundle(items=[law()]))
    schema = schema_of(model)
    for group, key in (("facts", "termination_agreement"), ("elements", "rent_after_end")):
        options = branches(schema, slot(schema, group, key))
        assert len(options) > 1
        for branch in options:
            assert list(branch["properties"])[0] == "reason"
            assert branch["additionalProperties"] is False


def test_not_stated_and_not_addressed_have_no_quote_field():
    """Spike 001: a separate `missing` with quotes allowed let models cite unrelated text."""
    _, model = assess(scripted(), bundle=EvidenceBundle(items=[law()]))
    schema = schema_of(model)
    for group, key, field, const in (("facts", "termination_agreement", "answer", "not_stated"),
                                     ("elements", "rent_after_end", "verdict", "not_addressed")):
        branch = next(b for b in branches(schema, slot(schema, group, key))
                      if b["properties"][field].get("const") == const)
        assert set(branch["properties"]) == {"reason", field}


# --- facts: code derives the status ------------------------------------------------------

@pytest.mark.parametrize("answer,quotes,status", [
    ("yes", ["합의했어요"], "known"),
    ("no", ["합의했어요"], "known"),
    ("not_stated", [], "missing"),
    ("conflicting", ["합의했어요", "동의하기 어렵습니다"], "conflicting"),
])
def test_fact_status_is_derived_from_the_answer(answer, quotes, status):
    output = scripted({"termination_agreement": said(answer, *quotes)}, rent_elements())
    result, _ = assess(output, message="합의했어요. 공제되나요?",
                       documents=[UserDocument(name="문자", text="동의하기 어렵습니다")])
    assert result.policy_input.assessment_status == "valid"
    assert result.policy_input.facts[0].status == status
    assert result.policy_input.facts[0].question  # the one question the policy may ask


def test_a_known_fact_keeps_which_way_the_user_answered():
    output = scripted({"termination_agreement": said("no", "합의는 안 받았고")}, rent_elements())
    result, _ = assess(output, message="합의는 안 받았고 먼저 나가요.")
    fact = result.assessment.facts[0]
    assert fact.status == "known" and fact.value == "없음"
    assert [o.quote for o in fact.observations] == ["합의는 안 받았고"]


def test_a_fact_written_only_in_an_attachment_is_seen_and_known():
    output = scripted({"termination_agreement": said("yes", "종료하기로 합의한다")}, rent_elements())
    result, model = assess(output, message="첨부대로 정리했어요.",
                           documents=[UserDocument(name="메모", text="2027-03-31에 종료하기로 합의한다.")])
    assert result.policy_input.facts[0].status == "known"
    prompt = "\n".join(m.content for m in model.calls[0].messages)
    assert "종료하기로 합의한다" in prompt and "D1" in prompt


@pytest.mark.parametrize("quote,expected", [
    ("합의했어요", "valid"),
    ("합의했어요 ", "valid"),      # Spike 001: trailing whitespace is not a different quote
    ("합의하지 않았어요", "invalid"),  # not in what the user wrote
])
def test_observation_quotes_must_be_in_the_users_text(quote, expected):
    output = scripted({"termination_agreement": said("yes", quote)}, rent_elements())
    result, _ = assess(output, message="합의했어요. 공제되나요?")
    assert result.policy_input.assessment_status == expected


@pytest.mark.parametrize("fact", [
    {"reason": "진술", "answer": "yes", "observations": []},                # a statement needs its quote
    {"reason": "진술", "answer": "yes", "observations": ["임차인은 차임을"]},  # legal text, not user material
    {"reason": "진술", "answer": "not_stated", "observations": ["합의했어요"]},
])
def test_a_fact_answer_outside_its_shape_is_a_failed_assessment(fact):
    """Grammar-constrained backends cannot emit these; the others must still fail closed."""
    result, _ = assess(scripted({"termination_agreement": fact}, rent_elements()), message="합의했어요.",
                       bundle=EvidenceBundle(items=[law()]))
    assert result.policy_input.assessment_status == "invalid"


# --- legal elements: code decides sufficiency --------------------------------------------

def test_evidence_is_supported_only_when_every_element_is_addressed():
    bundle = EvidenceBundle(items=[law("합의로 종료되면 그 뒤 차임 의무가 없다. 특약은 일방 퇴거에 적용된다.")])
    both = rent_elements(rent_after_end=addressed(("S1", "그 뒤 차임 의무가 없다")),
                         clause_scope=addressed(("S1", "특약은 일방 퇴거에 적용된다")))
    one = rent_elements(rent_after_end=addressed(("S1", "그 뒤 차임 의무가 없다")))
    facts = {"termination_agreement": said("yes", "합의했어요")}
    for elements, status in ((both, "supported"), (one, "insufficient")):
        result, _ = assess(scripted(facts, elements), message="합의했어요.", bundle=bundle)
        assert result.policy_input.assessment_status == "valid"
        assert result.policy_input.evidence[0].key == "deduction_basis"
        assert result.policy_input.evidence[0].status == status


def test_the_research_query_names_only_the_unaddressed_elements():
    s = module()
    bundle = EvidenceBundle(items=[law("합의로 종료되면 그 뒤 차임 의무가 없다.")])
    elements = rent_elements(rent_after_end=addressed(("S1", "그 뒤 차임 의무가 없다")))
    result, _ = assess(scripted({"termination_agreement": said("yes", "합의했어요")}, elements),
                       message="합의했어요.", bundle=bundle)
    proposition = result.assessment.evidence[0].proposition
    labels = dict(s.ISSUE_PROFILES["rent-deduction"].elements)
    assert labels["clause_scope"] in proposition and labels["rent_after_end"] not in proposition


def test_the_prompt_does_not_treat_every_general_rule_as_too_weak():
    """Smoke run: told that general principles never fill an element, EXAONE rejected a mandatory
    rule (Housing Lease Act art. 10) that states the element directly."""
    _, model = assess("not json")
    assert "일반 원칙" not in model.calls[0].messages[0].content


@pytest.mark.parametrize("support,expected", [
    (("S1", "차임을 지급하여야"), "valid"),
    (("S1", "원문에 없는 문장"), "invalid"),  # provenance failed: the whole assessment is unverifiable
    (("D1", "특약"), "invalid"),             # user material is never legal support
])
def test_a_legal_quote_must_be_in_a_legal_source(support, expected):
    output = scripted({}, {"agreement_effect": addressed(support)})
    result, _ = assess(output, issue="clause-explanation", bundle=EvidenceBundle(items=[law()]),
                       documents=[UserDocument(name="계약서", text="특약")])
    assert result.policy_input.assessment_status == expected


def test_an_addressed_element_needs_a_support():
    output = scripted({}, {"agreement_effect": {"reason": "다룸", "verdict": "addressed", "supports": []}})
    result, _ = assess(output, issue="clause-explanation", bundle=EvidenceBundle(items=[law()]))
    assert result.policy_input.assessment_status == "invalid"


def test_malformed_output_is_a_failed_assessment_not_a_gap():
    result, _ = assess("not json", issue="clause-explanation")
    assert result.policy_input.assessment_status == "invalid"


def test_the_prompt_gives_no_quotable_example_of_a_user_statement():
    """qwen3:8b and EXAONE both reported the prompt's example sentence as something the user said."""
    _, model = assess("not json")
    prompt = model.calls[0].messages[0].content
    for copied in ("'합의가 없었다'", "'별도 종료 합의 없음'"):
        assert copied not in prompt


def test_user_material_is_not_offered_as_legal_authority():
    _, model = assess("not json", documents=[UserDocument(name="계약서", text="특약")])
    user = json.loads(model.calls[0].messages[1].content)
    assert "D1" in user["user_observations"] and "D1" not in user["legal_sources"]


# --- case law: masking, roles, the whole document ----------------------------------------

def _case_bundle(role="기관·법원 판단", text="2026년 1월 1일 500만 원의 지급을 명했다."):
    return EvidenceBundle(items=[Evidence(source_id="S1", chunk_id="case1", document_id="case",
        document_type="court_decision", header={"Type": "Case", "Section role": role}, text=text, score=1.0)])


def _clause(quote, bundle, *, mask=False, document_text=None):
    return assess(scripted({}, {"agreement_effect": addressed(("S1", quote))}), issue="clause-explanation",
                  bundle=bundle, mask=mask, document_text=document_text)[0]


def _status(result):
    return result.policy_input.assessment_status, (result.policy_input.evidence[0].status
                                                    if result.policy_input.evidence else None)


def test_the_assessor_sees_the_same_case_figure_mask_as_the_generator():
    _, model = assess(scripted(), issue="clause-explanation", bundle=_case_bundle(), mask=True)
    prompt = "\n".join(m.content for m in model.calls[0].messages)
    assert "500만 원" not in prompt and "2026년 1월 1일" not in prompt and "(금액)" in prompt


@pytest.mark.parametrize("quote,mask,expected", [
    ("(날짜) (금액)의 지급을 명했다", True, "valid"),     # exactly what the model was shown
    ("2026년 1월 1일 500만 원의 지급", True, "invalid"),  # figures the model never saw
    ("2026년 1월 1일 500만 원의 지급", False, "valid"),   # unmasked runs check the original
])
def test_case_quotes_are_checked_against_the_text_the_model_read(quote, mask, expected):
    assert _clause(quote, _case_bundle(), mask=mask).policy_input.assessment_status == expected


@pytest.mark.parametrize("role,evidence", [
    ("기관·법원 판단", "supported"), ("법령·관련 법리", "supported"),
    # Every operational case chunk is unlabelled; refusing unknown would switch case law off.
    ("역할 미상 (판단·주장 구분 미확인)", "supported"),
    ("당사자 주장 (채택 여부 미확인)", "insufficient"), ("사실관계", "insufficient"),
    ("주문·결론", "insufficient"), ("복수 역할 (본문 구분 확인 필요)", "insufficient"),
])
def test_a_quote_that_is_not_reasoning_leaves_its_element_unaddressed(role, evidence):
    """The quote is verified, it just is not legal support: the element stays open (research,
    then withhold) instead of failing the whole assessment."""
    result = _clause("차임을 지급할 의무", _case_bundle(role=role, text="임차인은 차임을 지급할 의무가 있다."))
    assert _status(result) == ("valid", evidence)


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


_DOC = ("1. 기초사실\n원고는 2020. 3. 1. 피고와 임대차계약을 체결하였다.\n"
        "2. 당사자의 주장\n원고는 중도 퇴거 후에는 차임을 지급할 의무가 없다고 주장한다.\n"
        "3. 판단\n임차인은 계약이 종료될 때까지 차임을 지급할 의무가 있다.\n")


def test_a_party_claim_found_through_the_whole_document_is_not_support():
    """The window starts mid-section, so the heading that names the speaker is in an earlier chunk."""
    at = _DOC.index("원고는 중도")
    result = _clause("차임을 지급할 의무가 없다", _windowed_case(_DOC, at, at + 40), document_text=_Documents(_DOC))
    assert _status(result) == ("valid", "insufficient")
    assert result.support_roles == {"S1": "party_claim"}


def test_the_courts_reasoning_found_through_the_whole_document_is_support():
    at = _DOC.index("임차인은")
    result = _clause("차임을 지급할 의무가 있다", _windowed_case(_DOC, at, len(_DOC)), document_text=_Documents(_DOC))
    assert _status(result) == ("valid", "supported")
    assert result.support_roles == {"S1": "finding"}


def test_the_quote_is_located_in_the_masked_document_the_model_read():
    at = _DOC.index("원고는 2020")
    bundle = _windowed_case(_DOC, at, _DOC.index("2. 당사자"))
    result = _clause("원고는 (날짜) 피고와", bundle, mask=True, document_text=_Documents(_DOC))
    assert result.support_roles == {"S1": "facts"}  # found despite the masked date; facts are not support
    assert _status(result) == ("valid", "insufficient")


def test_without_the_document_the_chunk_alone_decides_and_unknown_is_recorded():
    at = _DOC.index("임차인은")
    for documents in (None, _Documents(None)):
        result = _clause("차임을 지급할 의무가 있다", _windowed_case(_DOC, at, len(_DOC)), document_text=documents)
        assert _status(result) == ("valid", "supported")
        assert result.support_roles == {"S1": "unknown"}


def test_statutes_are_never_looked_up_or_role_checked():
    documents = _Documents("원고는 임차인은 차임을 지급하여야 한다고 주장한다.")
    result = _clause("차임을 지급하여야 한다", EvidenceBundle(items=[law()]), document_text=documents)
    assert _status(result) == ("valid", "supported")
    assert documents.calls == [] and result.support_roles == {}


def test_interpretations_get_the_same_claim_filter_as_cases():
    doc = "[질의요지]\n임차인이 중도 퇴거하면 차임 지급 의무가 없다고 볼 수 있는지?\n[회답]\n차임 지급 의무가 있습니다.\n"
    at = doc.index("임차인이")
    bundle = EvidenceBundle(items=[Evidence(source_id="S1", chunk_id="i1", document_id="i",
        document_type="legal_interpretation", header={"Type": "Interpretation"}, text=doc[at:at + 30], score=1.0)])
    result = _clause("차임 지급 의무가 없다", bundle, document_text=_Documents(doc))
    assert result.support_roles == {"S1": "party_claim"}


def test_one_blocked_quote_marks_the_source_but_a_reasoning_quote_still_supports():
    at = _DOC.index("2. 당사자")
    bundle = _windowed_case(_DOC, at, len(_DOC))
    output = scripted({}, {"agreement_effect": addressed(("S1", "차임을 지급할 의무가 있다"),
                                                         ("S1", "차임을 지급할 의무가 없다"))})
    result, _ = assess(output, issue="clause-explanation", bundle=bundle, document_text=_Documents(_DOC))
    assert result.support_roles == {"S1": "party_claim"}  # the record shows the blocking role
    assert _status(result) == ("valid", "supported")


# --- quotes cut by the grammar's length limit -------------------------------------------

_LONG = "임대차는 당사자 일방이 상대방에게 목적물을 사용, 수익하게 할 것을 약정하고 상대방이 이에 대하여 차임을 지급할 것을 약정함으로써 그 효력이 생긴다."


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
    assert len(quote) <= 80
    assert _clause(quote, EvidenceBundle(items=[law(_LONG)])).policy_input.assessment_status == expected


def test_the_role_of_a_bent_quote_is_read_at_its_exact_prefix():
    doc = _DOC + ("임차인이 계약기간 중에 중도 퇴거하더라도 계약이 당사자의 합의로 종료되지 않은 이상, "
                  "그 이후의 기간에 대한 차임을 지급할 의무는 계속 남는다고 보아야 한다.")
    start = doc.index("임차인이 계약기간")
    quote = doc[start:][:79] + "X"
    result = _clause(quote, _windowed_case(doc, start, len(doc)), document_text=_Documents(doc))
    assert _status(result) == ("valid", "supported")
    assert result.support_roles == {"S1": "finding"}

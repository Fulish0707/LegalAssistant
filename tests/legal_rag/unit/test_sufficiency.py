"""Two-step sufficiency assessment: notes first, then fixed choices read from the notes.

Step 1 (AssessmentNotes): for each fact question and legal element, the model writes what it
found and copies the sentences. Step 2 (AssessmentChoice): from those notes alone, it picks an
answer. Code derives known/missing and supported/insufficient and checks every quote.
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

RENT_ELEMENTS = ("agreed_rent_after_end", "agreed_clause_applies", "unilateral_rent_after_move", "unilateral_clause_applies")
AGREED = ("agreed_rent_after_end", "agreed_clause_applies")
UNILATERAL = ("unilateral_rent_after_move", "unilateral_clause_applies")


def fact_note(*quotes, note="사용자 자료에서 찾은 내용"):
    return {"notes": note, "quotes": list(quotes)}


def element_note(*supports, note="법률 자료가 말하는 내용"):
    out = {"notes": note}
    if supports:  # without legal sources the schema has no supports field at all
        out["supports"] = [{"source_id": sid, "quote": quote} for sid, quote in supports]
    return out


def notes(facts=None, elements=None, issue="rent-deduction"):
    if elements is None:
        keys = RENT_ELEMENTS if issue == "rent-deduction" else ("agreement_effect",)
        elements = {key: element_note() for key in keys}
    if facts is None:
        facts = {"termination_agreement": fact_note()} if issue == "rent-deduction" else {}
    return json.dumps({"facts": facts, "elements": elements}, ensure_ascii=False)


def choice(facts=None, elements=None, issue="rent-deduction"):
    if elements is None:
        keys = RENT_ELEMENTS if issue == "rent-deduction" else ("agreement_effect",)
        elements = {key: "not_covered" for key in keys}
    if facts is None:
        facts = {"termination_agreement": "not_stated"} if issue == "rent-deduction" else {}
    return json.dumps({"facts": facts, "elements": elements})


def law(text="임차인은 차임을 지급하여야 한다.", sid="S1"):
    return Evidence(source_id=sid, chunk_id=f"law-{sid}", document_id="law", document_type="statute",
                    header={"Type": "Statute"}, text=text, score=1.0)


def assess(step1, step2=None, *, message="공제되나요?", documents=(), bundle=None, issue="rent-deduction",
           document_text=None, mask=False):
    """``step1`` holds both notes; the assessor asks for them in two calls with separate material."""
    s = module()
    try:
        both = json.loads(step1)
        fact_notes, legal_notes = json.dumps({"facts": both["facts"]}), json.dumps({"elements": both["elements"]})
    except (ValueError, KeyError, TypeError):
        fact_notes = legal_notes = step1
    model = FakeChatModel({"FactNotes": [fact_notes], "LegalNotes": [legal_notes],
                           "AssessmentChoice": [step2 if step2 is not None else choice(issue=issue)]})
    result = s.SufficiencyAssessor(model, document_text=document_text).assess(
        message, [], list(documents), bundle if bundle is not None else EvidenceBundle(),
        issue_id=issue, matter_id="lease-1", mask_case_figures=mask)
    return result, model


def call(model, name):
    return next(c for c in model.calls if c.response_schema.__name__ == name)


def schema_of(model, name):
    return call(model, name).response_schema.model_json_schema()


def sent(model, name):
    return "\n".join(m.content for m in call(model, name).messages)


def resolve(schema, node):
    while "$ref" in node:
        node = schema["$defs"][node["$ref"].rsplit("/", 1)[1]]
    return node


def slot(schema, group, key):
    return resolve(schema, resolve(schema, schema["properties"][group])["properties"][key])


def _status(result):
    return result.policy_input.assessment_status, (result.policy_input.evidence[0].status
                                                    if result.policy_input.evidence else None)


# --- step 1: notes ----------------------------------------------------------------------

def test_every_step_fixes_its_keys_to_the_issue():
    _, model = assess(notes(), bundle=EvidenceBundle(items=[law()]))
    for name, groups in (("FactNotes", ["facts"]), ("LegalNotes", ["elements"]), ("AssessmentChoice", ["facts", "elements"])):
        schema = schema_of(model, name)
        assert list(schema["properties"]) == groups
        for group, keys in (("facts", ["termination_agreement"]), ("elements", list(RENT_ELEMENTS))):
            if group in groups:
                node = resolve(schema, schema["properties"][group])
                assert node["required"] == keys and node["additionalProperties"] is False


def test_clause_explanation_asks_no_fact_question():
    _, model = assess(notes(issue="clause-explanation"), issue="clause-explanation")
    assert [c.response_schema.__name__ for c in model.calls] == ["LegalNotes", "AssessmentChoice"]
    schema = schema_of(model, "LegalNotes")
    assert resolve(schema, schema["properties"]["elements"])["required"] == ["agreement_effect"]


def test_fact_notes_see_only_user_material_and_legal_notes_only_legal_sources():
    """Smoke run: given both, EXAONE answered the legal questions from the user's message."""
    _, model = assess(notes(), documents=[UserDocument(name="계약서", text="고유한 특약 문장")],
                      bundle=EvidenceBundle(items=[law("고유한 법령 문장이다.")]))
    assert "고유한 특약 문장" in sent(model, "FactNotes") and "고유한 법령 문장" not in sent(model, "FactNotes")
    assert "고유한 법령 문장" in sent(model, "LegalNotes") and "고유한 특약 문장" not in sent(model, "LegalNotes")
    assert "공제되나요" not in sent(model, "LegalNotes")


def test_notes_come_before_the_copied_sentences_and_carry_no_verdict():
    _, model = assess(notes(), bundle=EvidenceBundle(items=[law()]))
    assert list(slot(schema_of(model, "FactNotes"), "facts", "termination_agreement")["properties"]) == ["notes", "quotes"]
    assert list(slot(schema_of(model, "LegalNotes"), "elements", "agreed_rent_after_end")["properties"]) == ["notes", "supports"]


def test_a_legal_support_can_only_name_a_legal_source():
    bundle = EvidenceBundle(items=[law(sid="S1"), law(sid="S2")])
    _, model = assess(notes(), bundle=bundle, documents=[UserDocument(name="계약서", text="특약")])
    schema = schema_of(model, "LegalNotes")
    support = resolve(schema, slot(schema, "elements", "agreed_rent_after_end")["properties"]["supports"]["items"])
    assert support["properties"]["source_id"]["enum"] == ["S1", "S2"]
    assert "chunk_id" not in support["properties"]
    assert support["properties"]["quote"]["maxLength"] <= 80


def test_without_legal_sources_an_element_has_no_support_field():
    _, model = assess(json.dumps({"facts": {"termination_agreement": fact_note()},
                                  "elements": {key: {"notes": "자료 없음"} for key in RENT_ELEMENTS}}))
    assert list(slot(schema_of(model, "LegalNotes"), "elements", "agreed_rent_after_end")["properties"]) == ["notes"]


def test_the_prompts_give_no_quotable_example_of_a_user_statement():
    """qwen3:8b and EXAONE both reported the prompt's example sentence as something the user said."""
    s = module()
    for prompt in (s._FACT_INSTRUCTIONS, s._LEGAL_INSTRUCTIONS, s._CHOICE_INSTRUCTIONS):
        for copied in ("'합의가 없었다'", "'별도 종료 합의 없음'"):
            assert copied not in prompt


def test_the_prompt_does_not_treat_every_general_rule_as_too_weak():
    """Smoke run: told that general principles never fill an element, EXAONE rejected a mandatory
    rule (Housing Lease Act art. 10) that states the element directly."""
    assert "일반 원칙" not in module()._LEGAL_INSTRUCTIONS


def test_legal_elements_are_asked_as_questions_the_law_can_answer_either_way():
    """v2 held-out: asked whether a source "addresses" an element, EXAONE answered whether it said yes."""
    s = module()
    for profile in s.ISSUE_PROFILES.values():
        for _, question, _ in profile.elements:
            assert question.endswith("가")


# --- step 2: choices read from the notes ------------------------------------------------

def test_the_choice_step_has_only_fixed_answers():
    _, model = assess(notes(), bundle=EvidenceBundle(items=[law()]))
    schema = schema_of(model, "AssessmentChoice")
    assert set(slot(schema, "facts", "termination_agreement")["enum"]) == {"yes", "no", "not_stated", "conflicting"}
    assert set(slot(schema, "elements", "agreed_rent_after_end")["enum"]) == {"yes", "no", "not_covered"}


def test_the_choice_step_reads_the_notes_not_the_sources():
    note = "판결은 합의로 종료되면 그 뒤 차임 의무가 없다고 한다"
    step1 = notes(elements={key: element_note(note=note if key == "agreed_rent_after_end" else "없음")
                            for key in RENT_ELEMENTS})
    _, model = assess(step1, bundle=EvidenceBundle(items=[law("임차인은 차임을 지급하여야 하는 고유 문장이다.")]))
    assert note in sent(model, "AssessmentChoice")
    assert "고유 문장" not in sent(model, "AssessmentChoice")


def test_a_rule_with_an_exception_is_still_an_answer_in_the_choice_instructions():
    """Smoke run: legal notes stated the rule ("no rent after an agreed end, unless agreed otherwise")
    and the choice step called every such note not_covered."""
    s = module()
    assert "근거가 부족" not in s._CHOICE_INSTRUCTIONS
    assert "단서" in s._CHOICE_INSTRUCTIONS


def test_a_malformed_step_is_a_failed_assessment_not_a_gap():
    assert assess("not json", issue="clause-explanation")[0].policy_input.assessment_status == "invalid"
    assert assess("not json")[0].policy_input.assessment_status == "invalid"
    result, _ = assess(notes(issue="clause-explanation"), "not json", issue="clause-explanation")
    assert result.policy_input.assessment_status == "invalid"


# --- facts --------------------------------------------------------------------------------

@pytest.mark.parametrize("answer,quotes,status", [
    ("yes", ["합의했어요"], "known"),
    ("no", ["합의했어요"], "known"),
    ("not_stated", [], "missing"),
    ("not_stated", ["합의했어요"], "missing"),  # quotes kept for context; the answer decides
    ("conflicting", ["합의했어요", "동의하기 어렵습니다"], "conflicting"),
])
def test_fact_status_is_derived_from_the_answer(answer, quotes, status):
    result, _ = assess(notes({"termination_agreement": fact_note(*quotes)}),
                       choice({"termination_agreement": answer}), message="합의했어요. 공제되나요?",
                       documents=[UserDocument(name="문자", text="동의하기 어렵습니다")])
    assert result.policy_input.assessment_status == "valid"
    assert result.policy_input.facts[0].status == status
    assert result.policy_input.facts[0].question  # the one question the policy may ask


@pytest.mark.parametrize("answer,quotes", [
    ("yes", []),                      # a statement needs the sentence that states it
    ("no", []),
    ("conflicting", ["합의했어요"]),  # a conflict needs both sides
])
def test_an_answer_without_the_sentences_it_needs_fails(answer, quotes):
    result, _ = assess(notes({"termination_agreement": fact_note(*quotes)}),
                       choice({"termination_agreement": answer}), message="합의했어요.")
    assert result.policy_input.assessment_status == "invalid"


def test_a_known_fact_keeps_which_way_the_user_answered_and_where_it_was_said():
    result, _ = assess(notes({"termination_agreement": fact_note("합의는 안 받았고")}),
                       choice({"termination_agreement": "no"}), message="합의는 안 받았고 먼저 나가요.",
                       documents=[UserDocument(name="계약서", text="특약: 조기 퇴거 시 차임 부담")])
    fact = result.assessment.facts[0]
    assert fact.status == "known" and fact.value == "없음"
    assert [(o.locator, o.quote) for o in fact.observations] == [("M0", "합의는 안 받았고")]


def test_a_fact_written_only_in_an_attachment_is_seen_and_known():
    result, model = assess(notes({"termination_agreement": fact_note("종료하기로 합의한다")}),
                           choice({"termination_agreement": "yes"}), message="첨부대로 정리했어요.",
                           documents=[UserDocument(name="메모", text="2027-03-31에 종료하기로 합의한다.")])
    assert result.policy_input.facts[0].status == "known"
    assert "종료하기로 합의한다" in sent(model, "FactNotes") and "D1" in sent(model, "FactNotes")


@pytest.mark.parametrize("quote,expected", [
    ("합의했어요", "valid"),
    ("합의했어요 ", "valid"),         # trailing whitespace is not a different quote
    ('"합의했어요"', "valid"),        # smoke run: the sentence wrapped in quotation marks
    ("“합의했어요”", "valid"),
    ("합의하지 않았어요", "invalid"),  # not in what the user wrote
    ("임차인은 차임을", "invalid"),    # legal text, not user material
])
def test_fact_quotes_must_be_in_the_users_text(quote, expected):
    result, _ = assess(notes({"termination_agreement": fact_note(quote)}), choice({"termination_agreement": "yes"}),
                       message="합의했어요. 공제되나요?", bundle=EvidenceBundle(items=[law()]))
    assert result.policy_input.assessment_status == expected


# --- legal elements -----------------------------------------------------------------------

_BOTH = "합의로 종료되면 그 뒤 차임 의무가 없다. 특약은 일방 퇴거에 적용된다."


def _rent(fact, answers, supported=()):
    """answers: element key -> choice; supported: element keys whose note quotes the law."""
    step1 = notes({"termination_agreement": fact_note("합의했어요")},
                  {key: element_note(*([("S1", "특약은 일방 퇴거에 적용된다")] if key in supported else []))
                   for key in RENT_ELEMENTS})
    step2 = choice({"termination_agreement": fact}, {key: answers.get(key, "not_covered") for key in RENT_ELEMENTS})
    return assess(step1, step2, message="합의했어요.", bundle=EvidenceBundle(items=[law(_BOTH)]))[0]


@pytest.mark.parametrize("fact,answers,supported,expected", [
    # an agreed ending needs only the questions about agreed endings, answered either way
    ("yes", {"agreed_rent_after_end": "no", "agreed_clause_applies": "no"}, AGREED, "supported"),
    ("yes", {"agreed_rent_after_end": "yes", "agreed_clause_applies": "no"}, AGREED, "supported"),
    ("yes", {"agreed_rent_after_end": "no", "agreed_clause_applies": "not_covered"}, AGREED, "insufficient"),
    ("yes", {"unilateral_rent_after_move": "yes", "unilateral_clause_applies": "yes"}, UNILATERAL, "insufficient"),
    # a unilateral move-out needs only the questions about it
    ("no", {"unilateral_rent_after_move": "yes", "unilateral_clause_applies": "yes"}, UNILATERAL, "supported"),
    ("no", {"agreed_rent_after_end": "no", "agreed_clause_applies": "no"}, AGREED, "insufficient"),
    # unknown facts: every question must be answered
    ("not_stated", {"agreed_rent_after_end": "no", "agreed_clause_applies": "no"}, AGREED, "insufficient"),
    ("not_stated", dict.fromkeys(RENT_ELEMENTS, "yes"), RENT_ELEMENTS, "supported"),
])
def test_law_must_answer_the_questions_that_match_the_established_facts(fact, answers, supported, expected):
    """Smoke run: asked about "합의 종료 또는 일방 퇴거" at once, EXAONE called the user's own case unclear."""
    fact_quotes = [] if fact == "not_stated" else ["합의했어요"]
    step1 = notes({"termination_agreement": fact_note(*fact_quotes)},
                  {key: element_note(*([("S1", "특약은 일방 퇴거에 적용된다")] if key in supported else []))
                   for key in RENT_ELEMENTS})
    step2 = choice({"termination_agreement": fact}, {key: answers.get(key, "not_covered") for key in RENT_ELEMENTS})
    result, _ = assess(step1, step2, message="합의했어요.", bundle=EvidenceBundle(items=[law(_BOTH)]))
    assert _status(result) == ("valid", expected)


def test_an_answer_with_no_verified_legal_sentence_leaves_the_element_open():
    """"no" may mean "the source says no" or "found nothing": without a quote it supports nothing."""
    result = _rent("yes", {"agreed_rent_after_end": "no", "agreed_clause_applies": "no"}, ("agreed_rent_after_end",))
    assert _status(result) == ("valid", "insufficient")


def test_the_research_query_names_only_the_open_questions_that_apply():
    s = module()
    result = _rent("yes", {"agreed_rent_after_end": "no"}, ("agreed_rent_after_end",))
    questions = {key: question for key, question, _ in s.ISSUE_PROFILES["rent-deduction"].elements}
    proposition = result.assessment.evidence[0].proposition
    assert questions["agreed_clause_applies"] in proposition
    assert questions["agreed_rent_after_end"] not in proposition
    assert questions["unilateral_clause_applies"] not in proposition


@pytest.mark.parametrize("support,expected", [
    (("S1", "차임을 지급하여야"), "valid"),
    (("S1", "원문에 없는 문장"), "invalid"),  # provenance failed: the whole assessment is unverifiable
    (("D1", "특약"), "invalid"),             # user material is never legal support
])
def test_a_legal_quote_must_be_in_a_legal_source(support, expected):
    step1 = notes(elements={"agreement_effect": element_note(support)}, issue="clause-explanation")
    result, _ = assess(step1, choice(elements={"agreement_effect": "yes"}, issue="clause-explanation"),
                       issue="clause-explanation", bundle=EvidenceBundle(items=[law()]),
                       documents=[UserDocument(name="계약서", text="특약")])
    assert result.policy_input.assessment_status == expected


@pytest.mark.parametrize("placeholder", ["[없음]", "없음", "해당 없음", "N/A"])
def test_a_placeholder_in_place_of_a_quote_is_no_quote(placeholder):
    """Smoke run: EXAONE wrote "[없음]" as a supporting quote."""
    step1 = notes(elements={"agreement_effect": element_note(("S1", placeholder))}, issue="clause-explanation")
    result, _ = assess(step1, choice(elements={"agreement_effect": "yes"}, issue="clause-explanation"),
                       issue="clause-explanation", bundle=EvidenceBundle(items=[law()]))
    assert _status(result) == ("valid", "insufficient")
    step1 = notes({"termination_agreement": fact_note(placeholder)})
    result, _ = assess(step1, choice({"termination_agreement": "not_stated"}))
    assert result.policy_input.assessment_status == "valid"


def test_a_fabricated_quote_fails_even_under_not_covered():
    step1 = notes(elements={"agreement_effect": element_note(("S1", "원문에 없는 문장"))}, issue="clause-explanation")
    result, _ = assess(step1, issue="clause-explanation", bundle=EvidenceBundle(items=[law()]))
    assert result.policy_input.assessment_status == "invalid"


@pytest.mark.parametrize("quote,expected", [
    ("합의로 종료되면 … 특약은 일방 퇴거에 적용된다", "valid"),   # v2 held-out: joined with an ellipsis
    ("합의로 종료되면...특약은 일방 퇴거에 적용된다", "valid"),
    ("특약은 일방 퇴거에 적용된다 … 합의로 종료되면", "invalid"),   # out of order
    ("합의로 종료되면 … 없는 문장입니다", "invalid"),
    ("합의로 … 특약은 일방 퇴거에 적용된다", "invalid"),            # a fragment too short to anchor
])
def test_a_quote_joined_with_an_ellipsis_is_checked_piece_by_piece(quote, expected):
    step1 = notes(elements={"agreement_effect": element_note(("S1", quote))}, issue="clause-explanation")
    result, _ = assess(step1, choice(elements={"agreement_effect": "yes"}, issue="clause-explanation"),
                       issue="clause-explanation", bundle=EvidenceBundle(items=[law(_BOTH)]))
    assert result.policy_input.assessment_status == expected


# --- case law: masking, roles, the whole document ----------------------------------------

def _case_bundle(role="기관·법원 판단", text="2026년 1월 1일 500만 원의 지급을 명했다."):
    return EvidenceBundle(items=[Evidence(source_id="S1", chunk_id="case1", document_id="case",
        document_type="court_decision", header={"Type": "Case", "Section role": role}, text=text, score=1.0)])


def _clause(*quotes, bundle, mask=False, document_text=None):
    step1 = notes(elements={"agreement_effect": element_note(*(("S1", q) for q in quotes))}, issue="clause-explanation")
    return assess(step1, choice(elements={"agreement_effect": "yes"}, issue="clause-explanation"),
                  issue="clause-explanation", bundle=bundle, mask=mask, document_text=document_text)[0]


def test_the_assessor_sees_the_same_case_figure_mask_as_the_generator():
    _, model = assess(notes(issue="clause-explanation"), issue="clause-explanation", bundle=_case_bundle(), mask=True)
    prompt = sent(model, "LegalNotes")
    assert "500만 원" not in prompt and "2026년 1월 1일" not in prompt and "(금액)" in prompt


@pytest.mark.parametrize("quote,mask,expected", [
    ("(날짜) (금액)의 지급을 명했다", True, "valid"),     # exactly what the model was shown
    ("2026년 1월 1일 500만 원의 지급", True, "invalid"),  # figures the model never saw
    ("2026년 1월 1일 500만 원의 지급", False, "valid"),   # unmasked runs check the original
])
def test_case_quotes_are_checked_against_the_text_the_model_read(quote, mask, expected):
    assert _clause(quote, bundle=_case_bundle(), mask=mask).policy_input.assessment_status == expected


@pytest.mark.parametrize("role,evidence", [
    ("기관·법원 판단", "supported"), ("법령·관련 법리", "supported"),
    # Every operational case chunk is unlabelled; refusing unknown would switch case law off.
    ("역할 미상 (판단·주장 구분 미확인)", "supported"),
    ("당사자 주장 (채택 여부 미확인)", "insufficient"), ("사실관계", "insufficient"),
    ("주문·결론", "insufficient"), ("복수 역할 (본문 구분 확인 필요)", "insufficient"),
])
def test_a_quote_that_is_not_reasoning_leaves_its_element_open(role, evidence):
    result = _clause("차임을 지급할 의무", bundle=_case_bundle(role=role, text="임차인은 차임을 지급할 의무가 있다."))
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
    result = _clause("차임을 지급할 의무가 없다", bundle=_windowed_case(_DOC, at, at + 40), document_text=_Documents(_DOC))
    assert _status(result) == ("valid", "insufficient")
    assert result.support_roles == {"S1": "party_claim"}


def test_the_courts_reasoning_found_through_the_whole_document_is_support():
    at = _DOC.index("임차인은")
    result = _clause("차임을 지급할 의무가 있다", bundle=_windowed_case(_DOC, at, len(_DOC)), document_text=_Documents(_DOC))
    assert _status(result) == ("valid", "supported")
    assert result.support_roles == {"S1": "finding"}


def test_the_quote_is_located_in_the_masked_document_the_model_read():
    at = _DOC.index("원고는 2020")
    bundle = _windowed_case(_DOC, at, _DOC.index("2. 당사자"))
    result = _clause("원고는 (날짜) 피고와", bundle=bundle, mask=True, document_text=_Documents(_DOC))
    assert result.support_roles == {"S1": "facts"}  # found despite the masked date; facts are not support
    assert _status(result) == ("valid", "insufficient")


def test_without_the_document_the_chunk_alone_decides_and_unknown_is_recorded():
    at = _DOC.index("임차인은")
    for documents in (None, _Documents(None)):
        result = _clause("차임을 지급할 의무가 있다", bundle=_windowed_case(_DOC, at, len(_DOC)), document_text=documents)
        assert _status(result) == ("valid", "supported")
        assert result.support_roles == {"S1": "unknown"}


def test_statutes_are_never_looked_up_or_role_checked():
    documents = _Documents("원고는 임차인은 차임을 지급하여야 한다고 주장한다.")
    result = _clause("차임을 지급하여야 한다", bundle=EvidenceBundle(items=[law()]), document_text=documents)
    assert _status(result) == ("valid", "supported")
    assert documents.calls == [] and result.support_roles == {}


def test_interpretations_get_the_same_claim_filter_as_cases():
    doc = "[질의요지]\n임차인이 중도 퇴거하면 차임 지급 의무가 없다고 볼 수 있는지?\n[회답]\n차임 지급 의무가 있습니다.\n"
    at = doc.index("임차인이")
    bundle = EvidenceBundle(items=[Evidence(source_id="S1", chunk_id="i1", document_id="i",
        document_type="legal_interpretation", header={"Type": "Interpretation"}, text=doc[at:at + 30], score=1.0)])
    result = _clause("차임 지급 의무가 없다", bundle=bundle, document_text=_Documents(doc))
    assert result.support_roles == {"S1": "party_claim"}


def test_one_blocked_quote_marks_the_source_but_a_reasoning_quote_still_supports():
    at = _DOC.index("2. 당사자")
    result = _clause("차임을 지급할 의무가 있다", "차임을 지급할 의무가 없다",
                     bundle=_windowed_case(_DOC, at, len(_DOC)), document_text=_Documents(_DOC))
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
    assert _clause(quote, bundle=EvidenceBundle(items=[law(_LONG)])).policy_input.assessment_status == expected


def test_the_role_of_a_bent_quote_is_read_at_its_exact_prefix():
    doc = _DOC + ("임차인이 계약기간 중에 중도 퇴거하더라도 계약이 당사자의 합의로 종료되지 않은 이상, "
                  "그 이후의 기간에 대한 차임을 지급할 의무는 계속 남는다고 보아야 한다.")
    start = doc.index("임차인이 계약기간")
    quote = doc[start:][:79] + "X"
    result = _clause(quote, bundle=_windowed_case(doc, start, len(doc)), document_text=_Documents(doc))
    assert _status(result) == ("valid", "supported")
    assert result.support_roles == {"S1": "finding"}

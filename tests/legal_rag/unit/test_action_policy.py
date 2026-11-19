"""Controlled assessment inputs, not legal-quality or live retrieval evaluation."""

import json
from pathlib import Path

import pytest

from legal_rag.rag import action_policy as p


FIXTURE = json.loads((Path(__file__).parents[1] / "fixtures/action_policy/fictional_lease.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["id"])
def test_fixed_case_replay(case):
    for step in case["steps"]:
        state = p.PolicyInput.model_validate_json(json.dumps(step["assessment"]))
        before = state.model_dump_json()
        decision = p.decide(state)
        assert decision.action == step["expected"]["action"]
        assert decision.reason == step["expected"]["reason"]
        assert state.model_dump_json() == before
        assert p.decide(state) == decision



def test_sufficient_facts_and_evidence_allow_an_answer():
    state = p.PolicyInput(
        matter_id="fictional-lease-1", issue_id="rent-deduction",
        facts=(p.FactRequirement(key="termination_agreement", status="known"),),
        evidence=(p.EvidenceRequirement(key="deduction_basis", status="supported"),),
    )
    decision = p.decide(state)
    assert decision.action == "answer"
    assert decision.reason == "requirements_supported"


@pytest.mark.parametrize("status", ["missing", "conflicting"])
def test_blocking_fact_gets_one_question_before_more_retrieval(status):
    state = p.PolicyInput(
        matter_id="fictional-lease-1", issue_id="rent-deduction",
        facts=(p.FactRequirement(key="termination_agreement", status=status,
                                 question="임대인과 종료일에 따로 합의하셨나요?"),),
        evidence=(p.EvidenceRequirement(key="deduction_basis", status="insufficient"),),
    )
    decision = p.decide(state)
    assert decision.action == "clarify"
    assert decision.target_fact_key == "termination_agreement"
    assert decision.question == "임대인과 종료일에 따로 합의하셨나요?"


@pytest.mark.parametrize("resolved", [False, True])
def test_used_clarification_never_blocks_new_facts_but_prevents_reasking(resolved):
    state = p.PolicyInput(
        matter_id="fictional-lease-1", issue_id="rent-deduction",
        clarification_used=True,
        facts=(p.FactRequirement(key="termination_agreement",
                                 status="known" if resolved else "missing",
                                 question="종료일에 합의하셨나요?"),),
        evidence=(p.EvidenceRequirement(key="deduction_basis", status="supported"),),
    )
    decision = p.decide(state)
    assert decision.action == ("answer" if resolved else "withhold")
    if not resolved:
        assert decision.reason == "facts_unresolved"
        assert decision.unresolved_keys == ("termination_agreement",)


@pytest.mark.parametrize("status", ["insufficient", "conflicting"])
@pytest.mark.parametrize("used", [False, True])
def test_evidence_gap_allows_only_one_additional_search(status, used):
    state = p.PolicyInput(
        matter_id="fictional-lease-1", issue_id="rent-deduction", research_used=used,
        facts=(p.FactRequirement(key="termination_agreement", status="known"),),
        evidence=(p.EvidenceRequirement(key="deduction_basis", status=status),),
    )
    decision = p.decide(state)
    assert decision.action == ("withhold" if used else "research")
    assert decision.unresolved_keys == ("deduction_basis",)
    assert decision.reason == ("evidence_unresolved" if used else "evidence_gap")


@pytest.mark.parametrize("blocking", [False, True])
def test_only_outcome_changing_missing_facts_trigger_a_question(blocking):
    state = p.PolicyInput(
        matter_id="fictional-lease-1", issue_id="clause-explanation",
        facts=(
            p.FactRequirement(key="deposit_amount", status="missing", blocks_outcome=False),
            p.FactRequirement(key="termination_agreement", status="missing",
                              blocks_outcome=blocking, question="종료일에 합의하셨나요?"),
        ),
        evidence=(p.EvidenceRequirement(key="clause_meaning", status="supported"),),
    )
    decision = p.decide(state)
    assert decision.action == ("clarify" if blocking else "answer")
    if blocking:
        assert decision.target_fact_key == "termination_agreement"


@pytest.mark.parametrize("assessment_status", ["invalid", "unavailable"])
def test_failed_assessment_is_not_treated_as_missing_user_facts(assessment_status):
    state = p.PolicyInput(
        matter_id="fictional-lease-1", issue_id="rent-deduction",
        assessment_status=assessment_status,
        facts=(p.FactRequirement(key="termination_agreement", status="missing",
                                 question="종료일에 합의하셨나요?"),),
        evidence=(p.EvidenceRequirement(key="deduction_basis", status="supported"),),
    )
    decision = p.decide(state)
    assert decision.action == "withhold"
    assert decision.reason == "assessment_failed"


@pytest.mark.parametrize("problem", ["empty_evidence", "blank_question", "duplicate_facts", "duplicate_evidence"])
def test_incomplete_or_ambiguous_assessment_fails_closed(problem):
    facts = (p.FactRequirement(key="agreement", status="known"),)
    evidence = (p.EvidenceRequirement(key="basis", status="supported"),)
    if problem == "empty_evidence":
        evidence = ()
    elif problem == "blank_question":
        facts = (p.FactRequirement(key="agreement", status="missing", question="  "),)
    elif problem == "duplicate_facts":
        facts += (p.FactRequirement(key="agreement", status="missing", question="합의했나요?"),)
    else:
        evidence += (p.EvidenceRequirement(key="basis", status="insufficient"),)
    state = p.PolicyInput(matter_id="fictional-lease-1", issue_id="rent-deduction",
                          facts=facts, evidence=evidence)
    decision = p.decide(state)
    assert decision.action == "withhold"
    assert decision.reason == "assessment_failed"






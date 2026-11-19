"""Pure post-retrieval decisions over assessed requirements; not wired into chat.

Callers supply assessments, not raw retrieval scores. This module neither proves
legal sufficiency nor persists/reserves budgets; those are executor responsibilities.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class FactRequirement(_Frozen):
    key: str = Field(min_length=1)
    status: Literal["known", "missing", "conflicting"]
    question: str = ""
    blocks_outcome: bool = True


class EvidenceRequirement(_Frozen):
    key: str = Field(min_length=1)
    status: Literal["supported", "insufficient", "conflicting"]


class PolicyInput(_Frozen):
    matter_id: str = Field(min_length=1)
    issue_id: str = Field(min_length=1)
    facts: tuple[FactRequirement, ...]
    evidence: tuple[EvidenceRequirement, ...]
    clarification_used: bool = False
    research_used: bool = False
    assessment_status: Literal["valid", "invalid", "unavailable"] = "valid"


class AnswerDecision(_Frozen):
    action: Literal["answer"] = "answer"
    reason: Literal["requirements_supported"] = "requirements_supported"


class ClarifyDecision(_Frozen):
    action: Literal["clarify"] = "clarify"
    reason: Literal["blocking_fact"] = "blocking_fact"
    target_fact_key: str = Field(min_length=1)
    question: str = Field(min_length=1)


class WithholdDecision(_Frozen):
    action: Literal["withhold"] = "withhold"
    reason: Literal["facts_unresolved", "evidence_unresolved", "assessment_failed"]
    unresolved_keys: tuple[str, ...]


class ResearchDecision(_Frozen):
    action: Literal["research"] = "research"
    reason: Literal["evidence_gap"] = "evidence_gap"
    unresolved_keys: tuple[str, ...] = Field(min_length=1)


Decision = AnswerDecision | ClarifyDecision | ResearchDecision | WithholdDecision


def decide(state: PolicyInput) -> Decision:
    """Choose the next action without I/O or input mutation."""
    if (
        state.assessment_status != "valid"
        or not state.evidence
        or len({fact.key for fact in state.facts}) != len(state.facts)
        or len({item.key for item in state.evidence}) != len(state.evidence)
    ):
        return WithholdDecision(reason="assessment_failed", unresolved_keys=())
    unresolved = tuple(fact for fact in state.facts if fact.blocks_outcome and fact.status != "known")
    if unresolved:
        if state.clarification_used:
            return WithholdDecision(reason="facts_unresolved", unresolved_keys=tuple(f.key for f in unresolved))
        fact = unresolved[0]
        if not fact.question.strip():
            return WithholdDecision(reason="assessment_failed", unresolved_keys=(fact.key,))
        return ClarifyDecision(target_fact_key=fact.key, question=fact.question)
    gaps = tuple(item.key for item in state.evidence if item.status != "supported")
    if gaps:
        if state.research_used:
            return WithholdDecision(reason="evidence_unresolved", unresolved_keys=gaps)
        return ResearchDecision(unresolved_keys=gaps)
    return AnswerDecision()

"""The structured answer contract the LLM must return."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from .documents import DocumentType


class CitationKind(StrEnum):
    statute = "statute"
    case = "case"
    interpretation = "interpretation"


# Retrieval filters on the fine-grained document_type; evidence rendering and citation
# verification need only the coarse kind. Deriving one from the other keeps a new
# document type from silently classifying as an interpretation.
_CITATION_KINDS: dict[DocumentType, CitationKind] = {
    DocumentType.statute: CitationKind.statute,
    DocumentType.judgment: CitationKind.case,
    DocumentType.court_decision: CitationKind.case,
    DocumentType.decision: CitationKind.case,
    DocumentType.patent_trial_decision: CitationKind.case,
    DocumentType.constitutional_decision: CitationKind.case,
    DocumentType.legal_interpretation: CitationKind.interpretation,
    DocumentType.admin_interpretation: CitationKind.interpretation,
}


def citation_kind(document_type: DocumentType) -> CitationKind:
    return _CITATION_KINDS[document_type]


def document_types_for(kind: CitationKind) -> list[DocumentType]:
    return [dt for dt, mapped in _CITATION_KINDS.items() if mapped is kind]


class Confidence(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    kind: CitationKind
    reference: str  # "민법 제750조" or a case number as written by the model
    quote: str | None = None


class LegalAnswer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    summary: str
    legal_issues: list[str] = Field(default_factory=list)
    analysis: str
    possible_actions: list[str] = Field(default_factory=list)
    additional_information_needed: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    uncertainty: str = ""
    confidence: Confidence

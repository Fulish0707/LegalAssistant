"""Canonical legal document types shared by every subsystem."""

from __future__ import annotations

import hashlib
import datetime as _dt
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

MetadataValue = str | int | float | bool | None


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DocumentType(StrEnum):
    judgment = "judgment"
    legal_interpretation = "legal_interpretation"
    decision = "decision"
    patent_trial_decision = "patent_trial_decision"
    constitutional_decision = "constitutional_decision"
    statute = "statute"
    court_decision = "court_decision"
    admin_interpretation = "admin_interpretation"


class SourceRef(_Frozen):
    dataset: str  # e.g. "aihub_judgment"
    file: str  # path relative to the raw directory
    record_key: str  # natural key inside the file: case number, law@version, reference number
    url: str | None = None


class LegalDocumentBase(_Frozen):
    id: str  # "{document_type}:{sha256(dataset|record_key)[:16]}"
    document_type: DocumentType
    title: str
    text: str  # deterministic flat rendering of the structured fields
    jurisdiction: str = "KR"
    legal_domain: str | None = None
    source: SourceRef
    created_date: _dt.date | None = None
    effective_date: _dt.date | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ImportedDocument(LegalDocumentBase):
    """Flat external source; no invented article or decision structure."""

    format: Literal["external_processed_v1"] = "external_processed_v1"


# Statute hierarchy: law → chapter → article (조) → paragraph (항) → item (호)


class StatuteItem(_Frozen):
    number: str  # "제1호"
    text: str


class StatuteParagraph(_Frozen):
    number: str  # "제1항"
    text: str
    items: list[StatuteItem] = Field(default_factory=list)


class StatuteArticle(_Frozen):
    number: str  # "제750조", "제3조의2"
    title: str | None = None
    text: str = ""  # article body when there are no paragraphs
    paragraphs: list[StatuteParagraph] = Field(default_factory=list)
    chapter: str | None = None


class Statute(LegalDocumentBase):
    document_type: Literal[DocumentType.statute] = DocumentType.statute
    law_name: str
    version: str | None = None
    promulgation_date: _dt.date | None = None
    articles: list[StatuteArticle]


class DecisionSection(StrEnum):
    issues = "issues"
    facts = "facts"
    reasoning = "reasoning"
    holding = "holding"
    summary = "summary"


SECTION_LABELS_KO: dict[DecisionSection, str] = {
    DecisionSection.issues: "판시사항",
    DecisionSection.facts: "사실관계",
    DecisionSection.reasoning: "판단",
    DecisionSection.holding: "주문",
    DecisionSection.summary: "요지",
}


class CourtDecision(LegalDocumentBase):
    document_type: Literal[DocumentType.court_decision] = DocumentType.court_decision
    court: str
    case_number: str
    decision_date: _dt.date | None = None
    case_type: str | None = None
    sections: dict[DecisionSection, str]
    referenced_statutes: list[str] = Field(default_factory=list)
    referenced_cases: list[str] = Field(default_factory=list)


class AdminInterpretation(LegalDocumentBase):
    document_type: Literal[DocumentType.admin_interpretation] = DocumentType.admin_interpretation
    authority: str
    reference_number: str
    date: _dt.date | None = None
    subject: str
    question: str
    interpretation: str
    legal_basis: list[str] = Field(default_factory=list)


LegalDocument = Annotated[Statute | CourtDecision | AdminInterpretation, Field(discriminator="document_type")]


def make_document_id(document_type: DocumentType, dataset: str, record_key: str) -> str:
    digest = hashlib.sha256(f"{dataset}|{record_key}".encode("utf-8")).hexdigest()[:16]
    return f"{document_type.value}:{digest}"

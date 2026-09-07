"""Invariants every canonical document must satisfy before it enters the corpus."""

from __future__ import annotations

from dataclasses import dataclass

from legal_rag.schemas import AdminInterpretation, CourtDecision, LegalDocument, Statute


@dataclass(frozen=True)
class ValidationFailure:
    document_id: str
    reason: str


def validate(doc: LegalDocument) -> ValidationFailure | None:
    if not doc.title.strip() or not doc.text.strip():
        return ValidationFailure(doc.id, "empty title or text")
    if isinstance(doc, Statute):
        if not doc.articles:
            return ValidationFailure(doc.id, "statute without articles")
        if any(not article.text.strip() and not article.paragraphs for article in doc.articles):
            return ValidationFailure(doc.id, "statute article without text or paragraphs")
    if isinstance(doc, CourtDecision) and (not doc.case_number.strip() or not doc.sections):
        return ValidationFailure(doc.id, "decision without case number or sections")
    if isinstance(doc, AdminInterpretation) and (not doc.question.strip() or not doc.interpretation.strip()):
        return ValidationFailure(doc.id, "interpretation without question or answer")
    return None

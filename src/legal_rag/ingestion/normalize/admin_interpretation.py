"""Normalize administrative interpretations (``admin_interpretation_v1``)."""

from __future__ import annotations

from typing import Any

from legal_rag.schemas import AdminInterpretation, DocumentType, SourceRef, make_document_id
from legal_rag.textutil.dates import parse_korean_date

from .base import NormalizationError, as_list, as_text


def render_interpretation_text(subject: str, question: str, interpretation: str, legal_basis: list[str]) -> str:
    lines = [subject, "", "[질의]", question, "", "[회답]", interpretation]
    if legal_basis:
        lines += ["", "[관련 법령]", ", ".join(legal_basis)]
    return "\n".join(lines)


def normalize_admin_interpretation(payload: dict[str, Any], *, dataset: str, file: str) -> AdminInterpretation:
    authority = as_text(payload.get("authority"))
    reference_number = as_text(payload.get("reference_number"))
    question = as_text(payload.get("question"))
    interpretation = as_text(payload.get("interpretation"))
    for name, value in (("authority", authority), ("reference_number", reference_number), ("question", question), ("interpretation", interpretation)):
        if not value:
            raise NormalizationError(f"{file}: missing {name}")
    subject = as_text(payload.get("subject")) or question[:60]
    legal_basis = as_list(payload.get("legal_basis"))
    return AdminInterpretation(
        id=make_document_id(DocumentType.admin_interpretation, dataset, reference_number),
        title=f"{authority} {reference_number} {subject}".strip(),
        text=render_interpretation_text(subject, question, interpretation, legal_basis),
        legal_domain=as_text(payload.get("legal_domain")) or None,
        source=SourceRef(dataset=dataset, file=file, record_key=reference_number, url=as_text(payload.get("source_url")) or None),
        created_date=parse_korean_date(as_text(payload.get("date"))),
        authority=authority,
        reference_number=reference_number,
        date=parse_korean_date(as_text(payload.get("date"))),
        subject=subject,
        question=question,
        interpretation=interpretation,
        legal_basis=legal_basis,
    )

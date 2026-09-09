"""Normalize the AI Hub court judgment layout (``aihub_judgment_v1``).

Layout assumption (see design risk R1): ``info`` holds case metadata, and the
narrative sections live under ``mentionedItems.rqestObjet`` (청구취지),
``facts.bsisFacts`` (기초사실), ``dcss.courtDcss`` (법원의 판단) and
``close.cnclsns`` (주문). Values may be strings or lists of strings.
"""

from __future__ import annotations

from typing import Any

from legal_rag.schemas import SECTION_LABELS_KO, CourtDecision, DecisionSection, DocumentType, SourceRef, make_document_id
from legal_rag.textutil.dates import parse_korean_date
from legal_rag.textutil.refs import extract_case_refs, extract_statute_refs

from .base import NormalizationError, as_text

_CASE_FIELD_DOMAIN = {"1": "민사", "2": "형사", "3": "행정", "4": "가사", "5": "특허"}

_SECTION_SOURCES: tuple[tuple[DecisionSection, str, str], ...] = (
    (DecisionSection.issues, "mentionedItems", "rqestObjet"),
    (DecisionSection.facts, "facts", "bsisFacts"),
    (DecisionSection.reasoning, "dcss", "courtDcss"),
    (DecisionSection.holding, "close", "cnclsns"),
)

SECTION_ORDER = (
    DecisionSection.issues,
    DecisionSection.facts,
    DecisionSection.reasoning,
    DecisionSection.holding,
    DecisionSection.summary,
)


def render_decision_text(court: str, case_number: str, title: str, sections: dict[DecisionSection, str]) -> str:
    lines = [f"{court} {case_number} {title}".strip()]
    for section in SECTION_ORDER:
        if section in sections and sections[section]:
            lines.append("")
            lines.append(f"[{SECTION_LABELS_KO[section]}]")
            lines.append(sections[section])
    return "\n".join(lines)


def normalize_aihub_judgment(payload: dict[str, Any], *, dataset: str, file: str) -> CourtDecision:
    info = payload.get("info") or {}
    if not isinstance(info, dict):
        raise NormalizationError(f"{file}: 'info' must be an object")
    case_number = as_text(info.get("caseNo"))
    court = as_text(info.get("courtNm"))
    if not case_number:
        raise NormalizationError(f"{file}: missing info.caseNo")
    if not court:
        raise NormalizationError(f"{file}: missing info.courtNm")

    sections: dict[DecisionSection, str] = {}
    for section, outer, inner in _SECTION_SOURCES:
        container = payload.get(outer) or {}
        text = as_text(container.get(inner)) if isinstance(container, dict) else ""
        if text:
            sections[section] = text
    if not sections:
        raise NormalizationError(f"{file}: judgment {case_number} has no narrative sections")

    case_name = as_text(info.get("caseNm"))
    domain = as_text(info.get("detailField")) or _CASE_FIELD_DOMAIN.get(as_text(info.get("caseField"))) or None
    all_text = "\n".join(sections.values())
    referenced_statutes = [ref.render() for ref in extract_statute_refs(all_text)]
    referenced_cases = [case for case in extract_case_refs(all_text) if case != case_number]

    title = f"{court} {case_number} {case_name}".strip()
    return CourtDecision(
        id=make_document_id(DocumentType.court_decision, dataset, case_number),
        title=title,
        text=render_decision_text(court, case_number, case_name, sections),
        legal_domain=domain,
        source=SourceRef(dataset=dataset, file=file, record_key=case_number),
        created_date=parse_korean_date(as_text(info.get("judmnAdjuDe"))),
        court=court,
        case_number=case_number,
        decision_date=parse_korean_date(as_text(info.get("judmnAdjuDe"))),
        case_type=as_text(info.get("trailField")) or None,
        sections=sections,
        referenced_statutes=referenced_statutes,
        referenced_cases=referenced_cases,
        metadata={"case_name": case_name} if case_name else {},
    )

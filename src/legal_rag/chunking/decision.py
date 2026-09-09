"""Court decision chunking: one chunk per semantic section, windowed when long."""

from __future__ import annotations

from legal_rag.schemas import SECTION_LABELS_KO, Chunk, CourtDecision, DecisionSection

from .base import DECISION_MAX_CHARS
from .fallback import chunk_text_windows

SECTION_ORDER = (
    DecisionSection.issues,
    DecisionSection.facts,
    DecisionSection.reasoning,
    DecisionSection.holding,
    DecisionSection.summary,
)


def chunk_decision(doc: CourtDecision, corpus_version: str, max_chars: int = DECISION_MAX_CHARS) -> list[Chunk]:
    chunks: list[Chunk] = []
    date_label = doc.decision_date.isoformat() if doc.decision_date else "날짜 미상"
    for section in SECTION_ORDER:
        text = doc.sections.get(section)
        if not text:
            continue
        header = f"{doc.court} {doc.case_number} ({date_label}) — {SECTION_LABELS_KO[section]}"
        chunks.extend(
            chunk_text_windows(
                doc,
                text,
                header=header,
                path_prefix=f"sec:{section.value}",
                start_order=len(chunks),
                corpus_version=corpus_version,
                max_chars=max_chars,
                court=doc.court,
                case_number=doc.case_number,
                decision_date=doc.decision_date,
                section=section.value,
            )
        )
    return chunks

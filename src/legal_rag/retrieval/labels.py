"""Display labels for search hits whose chunk names no court, case, law or title.

The criminal and administrative imports keep court and case number only in the header
of a document's first chunk, so about two thirds of their chunks carry no identifying
metadata. A hit borrows its document's header instead. A case number counts only when
the header's 사건 label introduces it: first chunks also cite lower-court and precedent
numbers, and naming the wrong case in an evidence panel is worse than naming none.
"""

from __future__ import annotations

import re
from collections import defaultdict

from legal_rag.schemas import ScoredChunk
from legal_rag.textutil.refs import CASE_RE

from .store import VectorStore

_CHUNK_INDEX = re.compile(r"(chunk_)\d+$")
_COURT = re.compile(r"^[가-힣]+(?:법원|지원|재판소|위원회)$")
_CASE_LABEL = re.compile(r"사\s*건")
_LABEL_FILLER = " \t\n○"
_HEADER_LINES = 10
_NAME_CHARS = 40

_DOMAINS = {"criminal": "형사", "civil": "민사", "administrative": "행정", "ip": "지식재산"}
_KINDS = {
    "statute": "법령", "judgment": "판결", "court_decision": "판결", "decision": "결정",
    "patent_trial_decision": "특허심판 심결", "legal_interpretation": "법령해석",
    "admin_interpretation": "행정해석",
}


def first_chunk_id(chunk_id: str) -> str | None:
    """The id of the same document's first chunk, for imported ``..chunk_NNNN`` ids."""
    match = _CHUNK_INDEX.search(chunk_id)
    if match is None:
        return None
    return chunk_id[: match.start()] + "chunk_" + "0" * len(chunk_id[match.start() + 6 :])


def _collapse(line: str) -> str:
    """'서 울 북 부 지 방 법 원' -> '서울북부지방법원'; ordinary lines are only stripped."""
    line = line.strip()
    return line.replace(" ", "") if re.fullmatch(r"(?:\S ){2,}\S", line) else line


def parse_header(text: str) -> str | None:
    lines = [_collapse(line) for line in text.splitlines()[:_HEADER_LINES] if line.strip()]
    if not lines:
        return None
    court = lines[0] if _COURT.match(lines[0]) else None
    head = "\n".join(lines)
    case_number = name = None
    for label in _CASE_LABEL.finditer(head):
        window = head[label.end() : label.end() + 80]
        found = CASE_RE.search(window)
        if found and not window[: found.start()].strip(_LABEL_FILLER):
            case_number = found.group(0)
            name = window[found.end() :].split("\n", 1)[0].strip()[:_NAME_CHARS] or None
            break
    if case_number and not court and "헌" in case_number:
        court = "헌법재판소"
    if not (court or case_number):
        return None
    return " ".join(part for part in (court, case_number, name) if part)


def fallback_label(document_type: str, legal_domain: str | None) -> str:
    if document_type == "constitutional_decision":
        return "헌법재판소 결정"
    kind = _KINDS.get(document_type, "자료")
    domain = _DOMAINS.get(legal_domain or "")
    return f"{domain} {kind}" if domain else kind


def _needs_label(hit: ScoredChunk) -> bool:
    chunk = hit.chunk
    metadata = chunk.metadata
    return not (chunk.title.strip() or metadata.court or metadata.case_number or metadata.law_name)


def document_labels(hits: list[ScoredChunk], stores: dict[str, VectorStore]) -> dict[str, str]:
    """document_id -> label, only for hits that cannot name themselves."""
    wanted: dict[str, dict[str, ScoredChunk]] = defaultdict(dict)
    for hit in hits:
        if _needs_label(hit):
            wanted[hit.chunk.metadata.corpus_version].setdefault(hit.chunk.document_id, hit)

    labels: dict[str, str] = {}
    for version, by_document in wanted.items():
        first_ids = {doc: first_chunk_id(hit.chunk.id) for doc, hit in by_document.items()}
        store = stores.get(version)
        texts = store.texts_by_chunk_id([i for i in first_ids.values() if i]) if store else {}
        for document_id, hit in by_document.items():
            header = texts.get(first_ids[document_id] or "")
            labels[document_id] = (parse_header(header) if header else None) or fallback_label(
                hit.chunk.document_type.value, hit.chunk.metadata.legal_domain
            )
    return labels

"""Structure-aware chunking of imported documents.

The upstream preprocessor cut imported corpora into fixed windows: median 3,151
characters and 71% starting mid-sentence. Judged pairs show relevance falling with
length (shortest quartile 0.80 relevant, longest 0.64), and reference notes keep
saying the chunk "cuts off before the ruling".

This chunker walks the tree from :mod:`legal_rag.chunking.korean_structure` and emits
one chunk per section, packing small siblings together and splitting oversized leaves
at sentence boundaries. Every chunk opens with a header line naming the document and
the section path, because dense retrieval and the reranker only ever see chunk text.

Sizing choices (all options):

* ``target_chars=1200`` — a section subtree at or below this is emitted whole.
* ``max_chars=2000`` — hard ceiling. A leaf is split only when it exceeds this, but
  once split the windows are packed to ``target_chars``, not to the ceiling.
* ``merge_threshold=400`` — a packed group below this joins a neighbour when the
  result still fits ``max_chars``.
* No overlap. Overlap exists to rescue a boundary that cuts an argument in half;
  section boundaries are where the argument actually ends, and duplicating text
  inflates the index and lets one passage win several slots in the top-k.

Top-level sections are never merged with each other: an article is the citation unit
and 주문/이유/판단 are the document's own divisions. A chunk headed 제1조 that secretly
holds 제1조–제5조 is worse for retrieval than a short, honest one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re

from legal_rag.schemas import Chunk, ChunkMetadata, CitationKind, DocumentType, citation_kind
from legal_rag.schemas.documents import ImportedDocument

from .korean_structure import ParsedStructure, Section, parse_structure, split_korean_sentences

MAX_HEADER_CHARS = 200


@dataclass(frozen=True)
class ChunkingOptions:
    target_chars: int = 1200
    max_chars: int = 2000
    merge_threshold: int = 400

    def __post_init__(self) -> None:
        if not 0 < self.merge_threshold <= self.target_chars <= self.max_chars:
            raise ValueError("require 0 < merge_threshold <= target_chars <= max_chars")


@dataclass(frozen=True)
class ChunkingReport:
    parsed: bool
    kind: str | None = None
    reason: str | None = None


def _date(value: object) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10].replace(".", "-"))
    except ValueError:
        return None


_BARE_ENUMERATOR = re.compile(r"^(?:\(\d{1,2}\)|[①-⑮]|\d{1,2}\.)$")


def _render_child(child: Section) -> str:
    """A bare enumerator heading — ``(1)``, ``①`` — reads as a prefix, not as its own line."""
    nested = _render_body(child)
    if not nested:
        return child.heading
    separator = " " if _BARE_ENUMERATOR.match(child.heading) else "\n"
    return f"{child.heading}{separator}{nested}"


def _render_body(section: Section) -> str:
    parts = [section.body] if section.body else []
    parts.extend(_render_child(child) for child in section.children)
    return "\n".join(parts)


def _document_label(doc: ImportedDocument) -> str:
    title = (doc.title or "").strip()
    if not title:
        return doc.id
    return f"{doc.id} {title}"


class _Builder:
    def __init__(self, doc: ImportedDocument, corpus_version: str, options: ChunkingOptions, kind: str | None) -> None:
        self.doc = doc
        self.corpus_version = corpus_version
        self.options = options
        self.kind = kind
        self.chunks: list[Chunk] = []
        self._ids: set[str] = set()
        original = doc.metadata.get("original") if isinstance(doc.metadata, dict) else None
        self.original = original if isinstance(original, dict) else {}
        nested = self.original.get("metadata")
        self.nested = nested if isinstance(nested, dict) else {}
        self.label = _document_label(doc)

    # -- header ---------------------------------------------------------------

    def header(self, path: list[str]) -> str:
        if self.kind == "statute":
            line = " ".join([self.doc.title or self.doc.id, *path])
        elif path:
            line = f"{self.label} · " + " > ".join(path)
        else:
            line = self.label
        line = " ".join(line.split())
        return line if len(line) <= MAX_HEADER_CHARS else line[: MAX_HEADER_CHARS - 1] + "…"

    # -- chunk construction ---------------------------------------------------

    def _metadata(self, path: list[Section], headings: list[str]) -> ChunkMetadata:
        kind = self.doc.document_type
        authority = self.original.get("authority")
        article = next((s.label for s in path if s.label.startswith("제") and "조" in s.label), None)
        paragraph = next((s.label for s in path if s.label and s.label[0] in "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮"), None)
        return ChunkMetadata(
            original=self.original,
            dataset=self.doc.source.dataset,
            corpus_version=self.corpus_version,
            legal_domain=self.doc.legal_domain,
            law_name=self.doc.title if kind is DocumentType.statute else None,
            article=article,
            paragraph=paragraph,
            authority=authority if isinstance(authority, str) else None,
            court=authority if citation_kind(kind) is CitationKind.case and isinstance(authority, str) else None,
            case_number=self.nested.get("case_number"),
            decision_date=_date(self.nested.get("decision_date")),
            effective_date=_date(self.nested.get("effective_date")),
            section=" > ".join(headings) or None,
        )

    def _unique(self, path_id: str) -> str:
        candidate = f"{self.doc.id}#{path_id}"
        suffix = 2
        while candidate in self._ids:
            candidate = f"{self.doc.id}#{path_id}~{suffix}"
            suffix += 1
        self._ids.add(candidate)
        return candidate

    def add(self, *, path_id: str, headings: list[str], sections: list[Section], body: str) -> None:
        header = self.header(headings)
        body = body.strip()
        room = self.options.max_chars - len(header) - 1
        pieces = [body] if len(body) <= room else _split_body(body, header, self.options)
        for number, piece in enumerate(pieces, start=1):
            identifier = path_id if len(pieces) == 1 else f"{path_id}/w{number}"
            self.chunks.append(
                Chunk(
                    id=self._unique(identifier),
                    document_id=self.doc.id,
                    document_type=self.doc.document_type,
                    path=identifier,
                    order=len(self.chunks),
                    text=f"{header}\n{piece}" if piece else header,
                    title=self.doc.title,
                    metadata=self._metadata(sections, headings),
                )
            )


def _split_body(body: str, header: str, options: ChunkingOptions) -> list[str]:
    """Whole-sentence windows; a single sentence longer than the ceiling is hard-cut."""
    # The ceiling is what keeps the finished chunk (header + newline + piece) inside
    # max_chars, so it can never be widened; room only decides how full a window is packed.
    ceiling = max(options.max_chars - len(header) - 1, 1)
    room = min(max(options.target_chars - len(header) - 1, 200), ceiling)
    windows: list[str] = []
    current = ""
    for sentence in split_korean_sentences(body) or [body]:
        while len(sentence) > ceiling:
            if current:
                windows.append(current)
                current = ""
            windows.append(sentence[:ceiling])
            sentence = sentence[ceiling:]
        if current and len(current) + 1 + len(sentence) > room:
            windows.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        windows.append(current)
    return windows or [""]


def _pack(sizes: list[int], options: ChunkingOptions, header_cost: int) -> list[list[int]]:
    """Greedy groups of consecutive siblings, then absorb groups below the merge threshold."""
    groups: list[list[int]] = []
    total = 0
    for index, size in enumerate(sizes):
        if groups and total + size + header_cost <= options.target_chars:
            groups[-1].append(index)
            total += size
        else:
            groups.append([index])
            total = size
    merged: list[list[int]] = []
    for group in groups:
        size = sum(sizes[i] for i in group)
        previous = sum(sizes[i] for i in merged[-1]) if merged else 0
        if merged and size < options.merge_threshold and previous + size + header_cost <= options.max_chars:
            merged[-1].extend(group)
        else:
            merged.append(group)
    # The pass above only looks backwards; a small *first* group has nothing behind it.
    if len(merged) > 1 and sum(sizes[i] for i in merged[0]) < options.merge_threshold:
        joined = sum(sizes[i] for i in merged[0] + merged[1])
        if joined + header_cost <= options.max_chars:
            merged[:2] = [merged[0] + merged[1]]
    return merged


_OWN_BODY_LABEL = "본문"


def _emit(builder: _Builder, section: Section, ancestors: list[Section], path_id: str) -> None:
    options = builder.options
    path = ancestors + [section]
    headings = [s.heading for s in path]
    body = _render_body(section)
    header_cost = len(builder.header(headings)) + 1
    if header_cost + len(body) <= options.target_chars or not section.children:
        builder.add(path_id=path_id, headings=headings, sections=path, body=body)
        return

    items: list[tuple[str, str, Section | None]] = []
    if section.body:
        items.append((_OWN_BODY_LABEL, section.body, None))
    for child in section.children:
        items.append((child.label, _render_child(child), child))

    for group in _pack([len(text) for _, text, _ in items], options, header_cost):
        first, last = items[group[0]], items[group[-1]]
        if len(group) == 1 and first[2] is not None:
            _emit(builder, first[2], path, f"{path_id}.{group[0] + 1}")
            continue
        label = first[0] if len(group) == 1 else f"{first[0]}–{last[0]}"
        member_id = f"{path_id}.{group[0] + 1}" if len(group) == 1 else f"{path_id}.{group[0] + 1}-{group[-1] + 1}"
        builder.add(
            path_id=member_id,
            headings=headings + [label],
            sections=path,
            body="\n".join(text for _, text, _ in (items[i] for i in group)),
        )


def _fallback(builder: _Builder, options: ChunkingOptions) -> None:
    for number, window in enumerate(_split_body(builder.doc.text, builder.header([]), options), start=1):
        builder.add(path_id=f"win:{number:04d}", headings=[], sections=[], body=window)


def chunk_imported_document(
    doc: ImportedDocument, corpus_version: str, options: ChunkingOptions | None = None
) -> tuple[list[Chunk], ChunkingReport]:
    """Chunk one imported document, falling back to sentence windows when unparseable."""
    options = options or ChunkingOptions()
    parsed: ParsedStructure | None = parse_structure(doc.text, doc.document_type.value)
    builder = _Builder(doc, corpus_version, options, parsed.kind if parsed else None)
    if parsed is None:
        _fallback(builder, options)
        return builder.chunks, ChunkingReport(parsed=False, reason="no structure")
    for index, section in enumerate(parsed.sections, start=1):
        _emit(builder, section, [], f"sec:{index}")
    if not builder.chunks:
        builder = _Builder(doc, corpus_version, options, None)
        _fallback(builder, options)
        return builder.chunks, ChunkingReport(parsed=False, reason="empty tree")
    return builder.chunks, ChunkingReport(parsed=True, kind=parsed.kind)


__all__ = ["ChunkingOptions", "ChunkingReport", "chunk_imported_document"]

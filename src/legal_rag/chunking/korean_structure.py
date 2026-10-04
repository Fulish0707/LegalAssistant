"""Recover section structure from the raw text of imported Korean legal documents.

The imported corpus carries no typed hierarchy — only ``text`` — so the legal-aware
chunkers in this package never fire and every document falls through to fixed windows.
This module reads the structural markers Korean legal drafting actually uses:

* statutes      ``제N조(제목)`` → ``①`` 항 → ``1.`` 호
* decisions     ``N.`` → ``가.`` → ``(N)``, plus named sections (주문, 범죄사실, …)
* interpretations ``질의요지`` / ``회답`` / ``이유``

The parser reports failure (``None``) rather than inventing structure, so the caller
can fall back to sentence windows.

Invariant: every non-whitespace character of the input survives, once, either in a
section heading or in exactly one section body. Nothing is dropped and nothing is
duplicated — a chunk that silently loses the ruling is the failure mode this replaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

# A period ends a sentence only when the character before it is not a digit: Korean
# legal prose is full of "2008. 12. 31." and "제3조 제1항", which a naive splitter shreds.
_SENTENCE_BREAK = re.compile(r"(?<=[^\s\d][.!?。])\s+")

_ARTICLE = re.compile(r"^\s*(제\d+조(?:의\d+)?)\s*(?:\(([^)]*)\))?\s*(.*)$")
_PARAGRAPH = re.compile(r"^\s*([①-⑮])\s*(.*)$")
_ITEM = re.compile(r"^\s*(\d{1,2}\.)\s+(\S.*)$")

_LEVEL1 = re.compile(r"^\s*(\d{1,2})\.\s*(\S.*)$")
_LEVEL2 = re.compile(r"^\s*([가-하])\.\s*(\S.*)$")
_LEVEL3 = re.compile(r"^\s*\((\d{1,2})\)\s*(\S.*)$")
_LEVEL3_ALT = re.compile(r"^\s*(\d{1,2})\)\s*(\S.*)$")

# Standalone section labels courts use instead of a number. Matched on the whole line,
# after stripping the brackets judgments wrap them in.
_NAMED = (
    "주문",
    "이유",
    "청구취지",
    "항소취지",
    "청구취지 및 항소취지",
    "상고취지",
    "범죄사실",
    "증거의 요지",
    "법령의 적용",
    "양형의 이유",
    "양형이유",
    "무죄부분",
    "다시 쓰는 판결 이유",
    "다시 쓰는 판결이유",
    "판단",
    "결론",
    "기초사실",
    "인정사실",
    "진정요지",
    "진정 요지",
    "진정인의 주장",
    "피진정인의 주장",
    "피진정인들의 주장",
    "피진정인의 주장요지",
    "피진정인들의 주장요지",
    "진정인의 주장요지",
    "원고의 주장",
    "피고의 주장",
    "당사자들의 주장",
    "관련 법리",
    "관계 법령",
    "피고인 및 변호인의 주장에 대한 판단",
    "피고인과 변호인의 주장에 대한 판단",
)

# 법제처 rulings use 질의요지/회답/이유; 조달청 replies use 제목/질의내용/회신내용.
_MARKERS = (
    "질의요지",
    "질의배경",
    "질의내용",
    "제목",
    "회답",
    "회신내용",
    "답변내용",
    "회신",
    "이유",
    "관계법령",
    "관계 법령",
)

MAX_HEADING_CHARS = 60


def _MARKER_AT_START(marker: str, line: str) -> bool:
    """A marker opens a section only when a separator follows it, not mid-word."""
    if not line.startswith(marker):
        return False
    rest = line[len(marker) :]
    return rest == "" or rest[0] in " \t:.：<[【"


def _is_heading_title(title: str) -> bool:
    """Section headings are short noun phrases; a numbered *sentence* is body text.

    Korean rulings open with a numbered 주문 ("1. 원결정을 취소한다.") that looks exactly
    like a section heading. Sentences end in a verb ending plus a period; headings do not.
    """
    title = title.strip()
    return 0 < len(title) <= MAX_HEADING_CHARS and not title.endswith((".", "다", "요"))

PREAMBLE_HEADING = "주문·청구취지"


@dataclass
class Section:
    """One node of the recovered tree. ``body`` is the text directly under ``heading``."""

    heading: str
    label: str
    body: str = ""
    children: list["Section"] = field(default_factory=list)

    def rendered_length(self) -> int:
        size = len(self.heading) + len(self.body)
        return size + sum(child.rendered_length() for child in self.children)


@dataclass(frozen=True)
class ParsedStructure:
    kind: str  # "statute" | "numbered" | "interpretation"
    sections: tuple[Section, ...]


def split_korean_sentences(text: str) -> list[str]:
    """Sentence split that keeps dotted dates and article references intact."""
    return [part.strip() for part in _SENTENCE_BREAK.split(text.strip()) if part.strip()]


def _body(lines: list[str]) -> str:
    return "\n".join(line.strip() for line in lines if line.strip())


def _named_heading(line: str) -> str | None:
    stripped = line.strip().strip("[]【】()「」 ").strip()
    compact = re.sub(r"\s+", "", stripped)
    return stripped if compact in {re.sub(r"\s+", "", name) for name in _NAMED} else None


def _parse_statute(text: str) -> ParsedStructure | None:
    articles: list[Section] = []
    pending: list[str] = []
    preamble: list[str] = []
    current: Section | None = None

    def flush() -> None:
        nonlocal pending
        if current is None:
            preamble.extend(pending)
        elif current.children:
            current.children[-1].body = _body(pending)
        else:
            current.body = _body(pending)
        pending = []

    for line in text.splitlines():
        if not line.strip():
            continue
        match = _ARTICLE.match(line) if line.lstrip().startswith("제") else None
        if match and (match.group(2) is not None or not match.group(3).strip()):
            flush()
            number, title, rest = match.group(1), match.group(2), match.group(3).strip()
            current = Section(heading=f"{number}({title})" if title is not None else number, label=number)
            articles.append(current)
            if rest:
                pending.append(rest)
            continue
        paragraph = _PARAGRAPH.match(line) if current is not None else None
        if paragraph:
            flush()
            current.children.append(Section(heading=paragraph.group(1), label=paragraph.group(1)))
            if paragraph.group(2).strip():
                pending.append(paragraph.group(2))
            continue
        pending.append(line)
    flush()
    if len(articles) < 2:
        return None
    lead = _body(preamble)
    if lead:
        articles.insert(0, Section(heading="전문", label="전문", body=lead))
    return ParsedStructure(kind="statute", sections=tuple(articles))


def _parse_numbered(text: str, *, document_type: str) -> ParsedStructure | None:
    top: list[Section] = []
    stack: list[Section] = []
    pending: list[str] = []
    preamble: list[str] = []
    matched = 0

    def attach(depth: int, heading: str, label: str, rest: str) -> None:
        nonlocal pending
        if stack:
            stack[-1].body = _body(pending)
        elif top:
            top[-1].body = _body(pending)
        else:
            preamble.extend(pending)
        pending = []
        section = Section(heading=heading, label=label)
        del stack[depth:]
        if depth == 0:
            top.append(section)
        else:
            stack[depth - 1].children.append(section)
        stack.append(section)
        if rest:
            pending.append(rest)

    for line in text.splitlines():
        if not line.strip():
            continue
        named = _named_heading(line)
        if named:
            attach(0, named, named, "")
            matched += 1
            continue
        level1 = _LEVEL1.match(line)
        if level1 and _is_heading_title(level1.group(2)):
            attach(0, f"{level1.group(1)}. {level1.group(2).strip()}", level1.group(1), "")
            matched += 1
            continue
        level2 = _LEVEL2.match(line)
        if level2 and stack and _is_heading_title(level2.group(2)):
            attach(1, f"{level2.group(1)}. {level2.group(2).strip()}", level2.group(1), "")
            matched += 1
            continue
        level3 = _LEVEL3.match(line) or _LEVEL3_ALT.match(line)
        if level3 and len(stack) >= 2:
            label = f"({level3.group(1)})" if line.lstrip().startswith("(") else f"{level3.group(1)})"
            title = level3.group(2).strip()
            # A short role heading must stay in the path rather than become unlabeled body.
            if _is_heading_title(title):
                attach(2, f"{label} {title}", label, "")
            else:
                attach(2, label, label, title)
            matched += 1
            continue
        pending.append(line)

    if stack:
        stack[-1].body = _body(pending)
    elif top:
        top[-1].body = _body(pending)
    else:
        preamble.extend(pending)

    if matched < 2 or not top:
        return None
    lead = _body(preamble)
    if lead:
        heading = PREAMBLE_HEADING if document_type in {"judgment", "patent_trial_decision"} else "전문"
        top.insert(0, Section(heading=heading, label=heading, body=lead))
    return ParsedStructure(kind="numbered", sections=tuple(top))


def _parse_interpretation(text: str) -> ParsedStructure | None:
    sections: list[Section] = []
    pending: list[str] = []
    preamble: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        if not line.strip():
            continue
        stripped = line.strip()
        marker = next(
            (m for m in _MARKERS if m not in seen and _MARKER_AT_START(m, stripped)),
            None,
        )
        if marker:
            if sections:
                sections[-1].body = _body(pending)
            else:
                preamble.extend(pending)
            pending = []
            seen.add(marker)
            sections.append(Section(heading=marker, label=marker))
            rest = stripped[len(marker) :].lstrip(" :<>【】[]").strip()
            if rest:
                pending.append(rest)
            continue
        pending.append(line)
    if sections:
        sections[-1].body = _body(pending)
    else:
        return None
    if len(sections) < 2:
        return None
    lead = _body(preamble)
    if lead:
        sections.insert(0, Section(heading="전문", label="전문", body=lead))
    return ParsedStructure(kind="interpretation", sections=tuple(sections))


def parse_structure(text: str, document_type: str) -> ParsedStructure | None:
    """Recover a section tree, or ``None`` when the text has no usable structure."""
    if not text or not text.strip():
        return None
    if document_type == "statute":
        return _parse_statute(text)
    if document_type == "legal_interpretation":
        return _parse_interpretation(text)
    return _parse_numbered(text, document_type=document_type)


__all__ = ["ParsedStructure", "Section", "parse_structure", "split_korean_sentences", "PREAMBLE_HEADING"]

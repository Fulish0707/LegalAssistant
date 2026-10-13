"""Deterministic extraction of statute article and case-number references.

This is a leaf module: ingestion, retrieval, and verification all use it, so it
must not import anything else from ``legal_rag``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

LAW_SUFFIX = r"(?:법|령|규칙|조례|규정)"
_LAW_NAME = rf"(?P<law>[가-힣A-Za-z0-9·]+?{LAW_SUFFIX})"
_ARTICLE = r"제\s*(?P<art>\d+)\s*조(?:\s*의\s*(?P<sub>\d+))?"
_PARAGRAPH = r"(?:\s*제\s*(?P<para>\d+)\s*항)?"
STATUTE_WITH_LAW = re.compile(rf"{_LAW_NAME}\s*{_ARTICLE}{_PARAGRAPH}")
STATUTE_BARE = re.compile(rf"{_ARTICLE}{_PARAGRAPH}")

CASE_TYPES = (
    # 헌법재판소 types first: alternation takes the first match, and bare 헌 would stop at 헌마.
    "헌가|헌나|헌다|헌라|헌마|헌바|헌사|헌아|"
    "가합|가단|가소|고합|고단|고정|다카|"
    "다|도|두|누|나|라|마|노|오|모|르|므|드|느|브|스|수|무|어|우|주|구|카|타|허|후|헌|초|즈|재|하|바"
)
CASE_PATTERN = rf"(?<![0-9])\d{{2,4}}(?:{CASE_TYPES})\d{{1,7}}(?![0-9])"
CASE_RE = re.compile(CASE_PATTERN)
# Preserve the public compact regex used by existing retrieval tokenization and
# evaluation. Reference extraction additionally accepts horizontal spacing.
_CASE_SPACE = r"[ \t\u00a0\u3000]*"
_SPACED_CASE_RE = re.compile(rf"(?<![0-9])\d{{2,4}}{_CASE_SPACE}(?:{CASE_TYPES}){_CASE_SPACE}\d{{1,7}}(?![0-9])")


@dataclass(frozen=True)
class StatuteRef:
    law_name: str | None
    article: str  # "제750조", "제3조의2"
    paragraph: str | None  # "제2항"

    def render(self) -> str:
        return f"{self.law_name} {self.article}" if self.law_name else self.article


def _article_from(match: re.Match) -> str:
    article = f"제{match.group('art')}조"
    if match.group("sub"):
        article += f"의{match.group('sub')}"
    return article


def _paragraph_from(match: re.Match) -> str | None:
    return f"제{match.group('para')}항" if match.group("para") else None


def normalize_article(raw: str) -> str:
    match = STATUTE_BARE.search(raw)
    if not match:
        return re.sub(r"\s+", "", raw)
    return _article_from(match)


def extract_statute_refs(text: str) -> list[StatuteRef]:
    found: list[tuple[int, StatuteRef]] = []
    consumed: list[tuple[int, int]] = []
    for match in STATUTE_WITH_LAW.finditer(text):
        found.append((match.start(), StatuteRef(match.group("law"), _article_from(match), _paragraph_from(match))))
        consumed.append(match.span())
    for match in STATUTE_BARE.finditer(text):
        if any(start <= match.start() < end for start, end in consumed):
            continue
        found.append((match.start(), StatuteRef(None, _article_from(match), _paragraph_from(match))))
    found.sort(key=lambda item: item[0])
    seen: set[tuple[str | None, str, str | None]] = set()
    refs: list[StatuteRef] = []
    for _, ref in found:
        key = (ref.law_name, ref.article, ref.paragraph)
        if key not in seen:
            seen.add(key)
            refs.append(ref)
    return refs


def extract_case_refs(text: str) -> list[str]:
    seen: list[str] = []
    for match in _SPACED_CASE_RE.finditer(text):
        case = re.sub(r"[ \t\u00a0\u3000]+", "", match.group(0))
        if case not in seen:
            seen.append(case)
    return seen

"""Role of a quoted position in windowed case text: who is speaking at that point.

Operational case chunks are fixed windows: none begins at a heading and about half mix
roles, so a chunk-level label cannot say whether a quote is the court's reasoning or a
party's claim. This module reads the whole document up to the quote instead.

The rule is ``rule_v1`` of the labelled evaluation (2026-11-19, 160 sentences from
passages actually retrieved, blind double labelling by two AI annotators, no human legal
review). On the frozen 100-sentence holdout it labelled no party claim as support (0/40,
95% upper bound 8.8%), but only 85% of what it called support was reasoning or a legal
rule; the rest was the court's own facts or conclusions. Use it to *block* claims, not to
certify findings. ``unknown`` (58% decided) is allowed and must be recorded.
After the evaluation one stricter change was made: a 질의요지 heading marks a claim.
"""

from __future__ import annotations

import re

from .roles import section_role

# Roles a case passage may have and still back a legal proposition.
LEGAL_SUPPORT_ROLES = frozenset({"finding", "legal_rule", "unknown"})

_TOP = {"주문", "이유", "청구취지", "항소취지", "상고취지", "사건개요", "판단", "결론", "범죄사실", "증거의요지",
        "법령의적용", "양형의이유", "판시사항", "판결요지", "결정요지", "참조조문", "질의요지", "회답", "전문", "사건의개요"}
_INLINE_TOP = re.compile(r"(【[^】]{1,14}】|\[[^\]]{1,14}\])")
_LEVELS = [
    (1, re.compile(r"^\d{1,2}\.\s*(.+)$")),
    (2, re.compile(r"^[가-하]\.\s*(.+)$")),
    (3, re.compile(r"^\d{1,2}\)\s*(.+)$")),
    (4, re.compile(r"^[가-하]\)\s*(.+)$")),
    (5, re.compile(r"^\(\d{1,2}\)\s*(.+)$")),
    (6, re.compile(r"^\([가-하]\)\s*(.+)$")),
    (7, re.compile(r"^[①-⑮]\s*(.+)$")),
]
# Sentence ends, including sentences glued without a space ("…아니하다.제117조").
_SENTENCE_END = re.compile(r"(?<=[다음함됨임]\.)|(?<=[?!])|(?<=\.)(?=\s)|\n")
_CLAIM = re.compile(r"(라고|다고|이라고|는|며|면서)\s?주장(한다|하나|하고|하므로|하는바|하였다|하고 있다)"
                    r"|취지로 주장|다투고 있다|다투나|항변한다|항변하나")
_REJECT = re.compile(r"이유\s?없|받아들일\s?수\s?없|받아들이지|타당하지\s?않|배척|인정할\s?수\s?없|어렵다|인정되지")


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text).strip("【】[]():：")


def _heading(line: str) -> tuple[int, str] | None:
    line = line.strip()
    if not line:
        return None
    if _squash(line) in _TOP and len(line) <= 24:
        return 0, _squash(line)
    bracketed = re.match(r"^【\s*(.+?)\s*】$|^\[\s*(.+?)\s*\]$", line)
    if bracketed:
        return 0, _squash(next(g for g in bracketed.groups() if g))
    for level, pattern in _LEVELS:
        match = pattern.match(line)
        if match:
            title = match.group(1).strip()
            # A date ("가.1975.7.9.") or a masked figure is not a heading; neither is a sentence.
            if (title[:1].isdigit() or title.startswith(("(날짜)", "(금액)")) or len(title) > 25
                    or title.endswith(("다.", "다", "라.", "함.", ","))):
                return None
            return level, line
    return None


def _title_role(title: str) -> str:
    t = re.sub(r"\s+", "", title)
    if "판단" in t or re.search(r"(상고|항소)이유.*(에대하여|에관하여)", t):
        return "finding"
    if "증거" in t and "요지" in t:
        return "evidence"
    if "범죄사실" in t:
        return "facts"
    # 질의요지 (the asking agency's question) was added after the frozen evaluation; it only
    # moves unknown to blocked, so it can make the filter stricter, never looser.
    if "공소사실" in t or "질의요지" in t or re.search(r"(항소|상고)이유의?요지", t):
        return "party_claim"
    if "양형" in t and "이유" in t:
        return "finding"
    if re.search(r"(피진정인|진정인|참고인|관계인|피해자)(\d|들|의|진술|$)", t):
        return "party_claim"
    if t.strip("【】[]") == "이유":
        return "reset"
    return section_role([title])


def _heading_role(text: str, position: int) -> str:
    prefix = _INLINE_TOP.sub(
        lambda m: "\n" + m.group(1) + "\n" if _squash(m.group(1)) in _TOP else m.group(1), text[:position])
    stack: list[tuple[int, str]] = []
    for line in prefix.split("\n")[:-1]:  # lines wholly before the quoted line
        heading = _heading(line)
        if heading:
            stack = [h for h in stack if h[0] < heading[0]] + [heading]
    role = "unknown"
    for _, title in stack:
        found = _title_role(title)
        if found == "reset":
            role = "unknown"
        elif found != "unknown":
            role = found
    return role


def sentence_at(text: str, position: int) -> str:
    """The sentence of ``text`` that contains ``position``."""
    start = 0
    for end in _SENTENCE_END.finditer(text):
        if end.end() > position and text[start:end.end()].strip():
            return text[start:end.end()].strip()
        start = end.end()
    return text[start:].strip()


def quote_role(text: str, position: int, sentence: str) -> str:
    """Role at ``position``: finding, legal_rule, party_claim, facts, order, evidence,
    mixed, quoted (inside an open quotation) or unknown. Sentence cues outrank headings,
    because a judgment section restates claims before it rejects them."""
    if _CLAIM.search(sentence) and not _REJECT.search(sentence):
        return "party_claim"
    window = text[max(0, position - 400):position]
    if window.count("‘") + window.count("“") > window.count("’") + window.count("”"):
        return "quoted"
    return _heading_role(text, position)

"""Name the article each fixed-window statute chunk starts in.

The imported corpora cut a whole statute into ~3,500-character windows and leave
``metadata.article`` empty. A window that starts mid-article therefore carries nothing
that says which article its opening text belongs to: 지방자치법 chunk 13 opens with the
body of 제124조, whose heading sits at the end of chunk 12, so a correct citation of
"지방자치법 제124조" to that chunk was counted as a mismatch and stripped (2026-09-28).
Headings later in the window are in its text already; only the opening needs naming.

Windows can overlap backwards, so the previous chunk's last heading is not reliable. The
chunk's start is located in the document text instead, and the last article heading at or
before it names the article.
"""

from __future__ import annotations

from bisect import bisect_right
import re

# "제124조(지방자치단체의 장의 권한대행 등)" or "제5조 삭제 <2010. 1. 1.>" at a line start.
# Cross-references such as "제3조제2항에 따라" never open a line with a parenthesis.
_HEADING = re.compile(r"(?m)^[ \t]*(제\d+조(?:의\d+)?)[ \t]*(?:[(（]|삭제)")
# Addenda restart numbering at 제1조; their articles must not name the main text.
_ADDENDA = re.compile(r"(?m)^[ \t]*부[ \t]*칙")
_PROBE = 60


class StatuteArticleLocator:
    """Feed one statute's chunks in order; returns the article each one opens in."""

    def __init__(self, document_text: str) -> None:
        self.text = document_text
        addenda = _ADDENDA.search(document_text)
        end = addenda.start() if addenda else len(document_text)
        headings = [(m.start(1), m.group(1)) for m in _HEADING.finditer(document_text) if m.start(1) < end]
        self._positions = [position for position, _ in headings]
        self._articles = [article for _, article in headings]
        self._end = end
        self._search_from = 0

    def article(self, chunk_text: str) -> str | None:
        body = chunk_text.lstrip()
        start = self.text.find(body[:_PROBE], self._search_from)
        if start < 0:
            start = self.text.find(body[:_PROBE])
        if start < 0 or start >= self._end:
            return None
        self._search_from = start + 1  # window starts only move forward
        index = bisect_right(self._positions, start) - 1
        if index >= 0:
            return self._articles[index]
        # Nothing before the window (e.g. it opens with "제1장 총칙"): name its first article.
        index = bisect_right(self._positions, start + len(body) - 1) - 1
        return self._articles[0] if index >= 0 else None


__all__ = ["StatuteArticleLocator"]

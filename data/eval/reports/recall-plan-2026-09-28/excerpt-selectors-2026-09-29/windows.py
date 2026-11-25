#!/usr/bin/env python3
"""Label-free windows over the text Flask would show for each top-10 document.

Nothing here reads a judgment or a quote. For every distinct document in a question's captured top 10 it rebuilds the
text the Flask formatter works from (distinct chunk texts in rank order joined by newline-ellipsis-newline, called
``joined``), then cuts it into candidate windows a selector can score:

* sentences come from the evidence-budget script's ``sentence_spans`` and are cut per chunk, so a window never crosses a
  chunk separator;
* a sentence longer than the token budget is cut at whitespace into pieces that fit;
* consecutive sentences are packed into a window up to the token budget, so the cross-encoder reads the whole window:
  ``<s> query </s></s> window </s>`` must stay within the reranker's 512 tokens (measured with the model's own
  tokenizer, not estimated from characters);
* each window starts half a window after the previous one (in sentences), so windows overlap by about half, and the
  first window of the first chunk starts at character 0 (the current prefix is always a candidate).

``display_span`` widens a chosen window to the 1,200 characters Flask shows per document.
"""
from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
LIVE = BASE.parent / "live-flask-current-2026-09-29-rootvenv"
sys.path.insert(0, str(LIVE))

import analyze_priority1_evidence_budget as eb  # noqa: E402  (also puts <repo>/src on sys.path)

SEPARATOR = eb.SEPARATOR
CHAR_LIMIT = eb.CHAR_LIMIT
MAX_DOCUMENTS = eb.MAX_DOCUMENTS
MAX_TOKENS = 512
SPECIALS = 4          # <s> ... </s></s> ... </s> for a sentence pair
SAFETY = 6            # tokens held back when packing; every window is re-measured exactly afterwards


def doc_parts(case_id: str, ranked: list[dict], pool: dict) -> list[dict]:
    """Same document grouping as ``eb.build_case_docs``, keeping each chunk's offset inside ``joined``."""
    docs: dict = {}
    for position, hit in enumerate(ranked):
        text = (pool[(case_id, hit["chunk_id"])].get("text") or "").strip()
        if not text:
            continue
        document_id = hit.get("document_id") or ("hit", position)
        entry = docs.setdefault(document_id, {"document_id": document_id, "parts": [], "seen": set()})
        if text in entry["seen"]:
            continue
        entry["seen"].add(text)
        entry["parts"].append({"chunk_id": hit["chunk_id"], "text": text})
    result = []
    for rank, entry in enumerate(docs.values()):
        offset = 0
        for part in entry["parts"]:
            part["offset"] = offset
            offset += len(part["text"]) + len(SEPARATOR)
        result.append({"document_id": entry["document_id"], "rank": rank, "parts": entry["parts"],
                       "joined": SEPARATOR.join(part["text"] for part in entry["parts"])})
    return result


def split_long(text: str, start: int, end: int, budget: int, count) -> list[tuple[int, int, int]]:
    """Cut one over-budget sentence at whitespace into pieces of at most ``budget`` tokens."""
    pieces, cursor = [], start
    while cursor < end:
        limit = end
        while count(text[cursor:limit]) > budget:
            cut = text.rfind(" ", cursor + 1, limit)
            limit = cut if cut > cursor else cursor + max(1, (limit - cursor) // 2)
        pieces.append((cursor, limit, count(text[cursor:limit])))
        cursor = limit
        while cursor < end and text[cursor].isspace():
            cursor += 1
    return pieces


def part_windows(part: dict, budget: int, count) -> list[tuple[int, int]]:
    """Sentence-packed windows over one chunk, in ``joined`` coordinates."""
    text = part["text"]
    units: list[tuple[int, int, int]] = []
    for left, right in eb.sentence_spans(text):
        tokens = count(text[left:right])
        units.extend([(left, right, tokens)] if tokens <= budget else split_long(text, left, right, budget, count))
    windows, first = [], 0
    while first < len(units):
        last, total = first, 0
        while last < len(units) and total + units[last][2] <= budget:
            total += units[last][2]
            last += 1
        last = max(last, first + 1)
        windows.append((part["offset"] + units[first][0], part["offset"] + units[last - 1][1]))
        if last >= len(units):
            break
        first += max(1, (last - first) // 2)
    return windows


def build_windows(docs: list[dict], query_tokens: int, count) -> None:
    """Attach ``windows`` (start, end in ``joined``) to every document."""
    budget = MAX_TOKENS - SPECIALS - SAFETY - query_tokens
    if budget < 64:
        raise ValueError(f"query leaves only {budget} tokens for a window")
    for doc in docs:
        doc["windows"] = [span for part in doc["parts"] for span in part_windows(part, budget, count)]


def display_span(start: int, end: int, length: int, limit: int = CHAR_LIMIT) -> tuple[int, int]:
    """Widen [start, end) evenly on both sides to ``limit`` characters, kept inside [0, length)."""
    if length <= limit:
        return 0, length
    if end - start >= limit:
        return start, start + limit
    extra = limit - (end - start)
    new_start, new_end = start - extra // 2, end + (extra - extra // 2)
    if new_start < 0:
        new_start, new_end = 0, limit
    elif new_end > length:
        new_start, new_end = length - limit, length
    return new_start, new_end

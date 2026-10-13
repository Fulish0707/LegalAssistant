"""Statute articles: the ones a case text cites, and a law's own articles as stored."""

from __future__ import annotations

import re
from itertools import groupby

from legal_rag.schemas import Chunk, ScoredChunk
from legal_rag.textutil.refs import STATUTE_BARE, StatuteRef, _article_from, _paragraph_from

from .document import full_text
from .store import VectorStore

# The word a citation leans on: "주택임대차보호법 제3조", "같은 법 제3조", "법 제3조".
_LAW_WORD = re.compile(r"([가-힣A-Za-z0-9·]*(?:법|령|규칙|조례|규정))$")
# These point back at the law named before them.
_SAME_LAW = ("법", "같은법", "동법", "위법", "이법", "법률", "규정")
_NAME_WINDOW = 60


def compact_names(names) -> dict[str, str]:
    """Law names keyed without spaces: a judgment writes 상가건물임대차보호법 for 상가건물 임대차보호법."""
    return {name.replace(" ", ""): name for name in names}


def cited_articles(text: str, laws: dict[str, str]) -> list[StatuteRef]:
    """Each article of a known law that the text cites, once, in the order first cited.

    `laws` is `compact_names` of the laws that exist. A citation takes the longest known name that
    ends just before it. One without a name of its own (같은 법 제3조, 제3조 제2항) belongs to the law
    named before it. A name that is no known law (호법, a repealed act) is dropped, and so are the
    bare citations after it, which belong to it.
    """
    text = text.replace("「", " ").replace("」", " ")
    cited: dict[StatuteRef, None] = {}
    law = None
    for match in STATUTE_BARE.finditer(text):
        words = text[max(0, match.start() - _NAME_WINDOW):match.start()].split()
        word = _LAW_WORD.search(words[-1]) if words else None
        if word and word.group(1) not in _SAME_LAW:
            before = "".join(words)  # a name may be written with or without its spaces
            law = next((laws[before[-size:]] for size in range(len(before), 1, -1) if before[-size:] in laws), None)
        if law:
            cited[StatuteRef(law, _article_from(match), _paragraph_from(match))] = None
    return list(cited)

# A chunk opens with "<law> 제3조(대항력 등)"; a paragraph chunk adds " 제②항".
_TITLE = re.compile(r"제\d+조(?:의\d+)?\(([^)\n]*)\)")


def statutes_cited(hits: list[ScoredChunk], stores: dict[str, VectorStore], laws: dict[str, str]) -> dict[str, list[StatuteRef]]:
    """document_id -> the articles the whole document cites, for each hit's document.

    The whole text is read, not the passage that matched: a judgment names a law once and then
    cites its articles bare for pages.
    """
    cited = {}
    for hit in hits:
        chunk = hit.chunk
        store = stores.get(chunk.metadata.corpus_version)
        text = full_text(store.document_chunks(chunk), chunk.id)[0] if store else chunk.text
        cited[chunk.document_id] = cited_articles(text, laws)
    return cited


def articles(chunks: list[Chunk]) -> list[dict]:
    """[{article, title, text, chunk_id}] in the law's order.

    A long article is stored as one chunk per paragraph. Its paragraphs are joined under the
    paragraph label each header carries, as a short article already reads in its single chunk.
    """
    listed = []
    for number, group in groupby(chunks, key=lambda chunk: chunk.metadata.article):
        parts = list(group)
        header = parts[0].text.split("\n", 1)[0]
        title = _TITLE.search(header)
        bodies = []
        for part in parts:
            body = part.text.split("\n", 1)[-1].strip() if "\n" in part.text else ""
            bodies.append(f"{part.metadata.paragraph} {body}" if part.metadata.paragraph else body)
        listed.append({"article": number, "title": title.group(1) if title else "",
                       "text": "\n".join(body for body in bodies if body), "chunk_id": parts[0].id})
    return listed

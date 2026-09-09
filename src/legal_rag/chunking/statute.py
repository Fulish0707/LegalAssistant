"""Statute chunking: article when it fits, otherwise article + paragraph, then windows."""

from __future__ import annotations

from legal_rag.schemas import Chunk, Statute, StatuteArticle, StatuteParagraph

from .base import STATUTE_MAX_CHARS, make_chunk
from .fallback import chunk_text_windows


def _article_header(law_name: str, article: StatuteArticle) -> str:
    return f"{law_name} {article.number}" + (f"({article.title})" if article.title else "")


def _paragraph_body(paragraph: StatuteParagraph) -> str:
    lines = [paragraph.text.strip()] if paragraph.text.strip() else []
    lines += [f"{item.number} {item.text}".strip() for item in paragraph.items]
    return "\n".join(lines)


def _article_body(article: StatuteArticle) -> str:
    lines = [article.text.strip()] if article.text.strip() else []
    for paragraph in article.paragraphs:
        lines.append(f"{paragraph.number} {_paragraph_body(paragraph)}".strip())
    return "\n".join(lines)


def chunk_statute(doc: Statute, corpus_version: str, max_chars: int = STATUTE_MAX_CHARS) -> list[Chunk]:
    chunks: list[Chunk] = []
    for article in doc.articles:
        header = _article_header(doc.law_name, article)
        body = _article_body(article)
        path = f"art:{article.number}"
        meta = {"law_name": doc.law_name, "article": article.number}
        if len(header) + 1 + len(body) <= max_chars:
            chunks.append(make_chunk(doc, path=path, order=len(chunks), text=f"{header}\n{body}", corpus_version=corpus_version, **meta))
            continue
        if not article.paragraphs:
            chunks.extend(chunk_text_windows(doc, body, header=header, path_prefix=path, start_order=len(chunks), corpus_version=corpus_version, max_chars=max_chars, **meta))
            continue
        for paragraph in article.paragraphs:
            para_header = f"{header} {paragraph.number}"
            chunks.extend(
                chunk_text_windows(
                    doc,
                    _paragraph_body(paragraph),
                    header=para_header,
                    path_prefix=f"{path}/para:{paragraph.number}",
                    start_order=len(chunks),
                    corpus_version=corpus_version,
                    max_chars=max_chars,
                    paragraph=paragraph.number,
                    **meta,
                )
            )
    return chunks

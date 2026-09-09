"""Normalize the internal statute layout (``statute_json_v1``).

Layout assumption (design decision D1): a law.go.kr-like tree of
``articles[] → paragraphs[] → items[]`` with Korean numbering strings.
"""

from __future__ import annotations

import re
from typing import Any

from legal_rag.schemas import DocumentType, SourceRef, Statute, StatuteArticle, StatuteItem, StatuteParagraph, make_document_id
from legal_rag.textutil.dates import parse_korean_date
from legal_rag.textutil.refs import normalize_article

from .base import NormalizationError, as_text


def _number(raw: Any, suffix: str) -> str:
    text = re.sub(r"\s+", "", as_text(raw))
    if not text:
        return ""
    if text.isdigit():
        return f"제{text}{suffix}"
    return text


def render_article(article: StatuteArticle) -> str:
    head = article.number + (f"({article.title})" if article.title else "")
    lines = [f"{head} {article.text}".rstrip()]
    for paragraph in article.paragraphs:
        lines.append(f"  {paragraph.number} {paragraph.text}".rstrip())
        for item in paragraph.items:
            lines.append(f"    {item.number} {item.text}".rstrip())
    return "\n".join(lines)


def render_statute_text(law_name: str, articles: list[StatuteArticle]) -> str:
    return "\n".join([law_name, *[render_article(article) for article in articles]])


def normalize_statute_json(payload: dict[str, Any], *, dataset: str, file: str) -> Statute:
    law_name = as_text(payload.get("law_name"))
    if not law_name:
        raise NormalizationError(f"{file}: missing law_name")
    raw_articles = payload.get("articles")
    if not isinstance(raw_articles, list) or not raw_articles:
        raise NormalizationError(f"{file}: statute {law_name} has no articles")

    articles: list[StatuteArticle] = []
    for raw in raw_articles:
        if not isinstance(raw, dict):
            continue
        number = normalize_article(as_text(raw.get("number")))
        if not number:
            raise NormalizationError(f"{file}: article without number in {law_name}")
        paragraphs = []
        for raw_para in raw.get("paragraphs") or []:
            items = [
                StatuteItem(number=_number(raw_item.get("number"), "호"), text=as_text(raw_item.get("text")))
                for raw_item in raw_para.get("items") or []
                if isinstance(raw_item, dict)
            ]
            paragraphs.append(
                StatuteParagraph(number=_number(raw_para.get("number"), "항"), text=as_text(raw_para.get("text")), items=items)
            )
        articles.append(
            StatuteArticle(
                number=number,
                title=as_text(raw.get("title")) or None,
                text=as_text(raw.get("text")),
                paragraphs=paragraphs,
                chapter=as_text(raw.get("chapter")) or None,
            )
        )

    version = as_text(payload.get("version")) or None
    record_key = f"{law_name}@{version or 'unversioned'}"
    return Statute(
        id=make_document_id(DocumentType.statute, dataset, record_key),
        title=law_name,
        text=render_statute_text(law_name, articles),
        legal_domain=as_text(payload.get("legal_domain")) or None,
        source=SourceRef(dataset=dataset, file=file, record_key=record_key, url=as_text(payload.get("source_url")) or None),
        effective_date=parse_korean_date(as_text(payload.get("effective_date"))),
        law_name=law_name,
        version=version,
        promulgation_date=parse_korean_date(as_text(payload.get("promulgation_date"))),
        articles=articles,
    )

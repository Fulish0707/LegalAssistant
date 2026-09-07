"""Translate ``RetrievalFilters`` into store-neutral conditions applied at query time."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from legal_rag.schemas import RetrievalFilters


@dataclass(frozen=True)
class MatchAny:
    key: str
    values: list[str]


@dataclass(frozen=True)
class Range:
    key: str
    gte: int | None = None
    lte: int | None = None


Condition = MatchAny | Range


def date_int(value: date | None) -> int | None:
    return None if value is None else value.year * 10000 + value.month * 100 + value.day


def to_conditions(filters: RetrievalFilters | None) -> list[Condition]:
    if filters is None:
        return []
    conditions: list[Condition] = []
    if filters.document_types:
        conditions.append(MatchAny("document_type", [dt.value for dt in filters.document_types]))
    if filters.legal_domain:
        conditions.append(MatchAny("metadata.legal_domain", [filters.legal_domain]))
    if filters.court:
        conditions.append(MatchAny("metadata.court", [filters.court]))
    if filters.law_name:
        conditions.append(MatchAny("metadata.law_name", [filters.law_name]))
    if filters.decided_after or filters.decided_before:
        conditions.append(Range("decision_date_int", gte=date_int(filters.decided_after), lte=date_int(filters.decided_before)))
    if filters.effective_on:
        conditions.append(Range("effective_date_int", lte=date_int(filters.effective_on)))
    return conditions

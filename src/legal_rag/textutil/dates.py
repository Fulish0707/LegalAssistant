"""Parse the date spellings found in Korean legal datasets."""

from __future__ import annotations

import re
from datetime import date

_PATTERNS = (
    re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$"),
    re.compile(r"^(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.?$"),
    re.compile(r"^(\d{4})(\d{2})(\d{2})$"),
    re.compile(r"^(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일$"),
)


def parse_korean_date(value: str | None) -> date | None:
    if not value:
        return None
    text = str(value).strip()
    for pattern in _PATTERNS:
        match = pattern.match(text)
        if match:
            year, month, day = (int(group) for group in match.groups())
            try:
                return date(year, month, day)
            except ValueError:
                return None
    return None

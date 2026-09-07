"""Sentence splitting tuned for Korean legal prose."""

import re

_SPLIT = re.compile(r"(?<=[.!?。])\s+")


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SPLIT.split(text.strip()) if part.strip()]

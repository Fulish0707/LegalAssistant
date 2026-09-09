"""Tokenizers for the sparse (BM25) leg.

``RegexTokenizer`` needs no models: it keeps legal identifiers whole
(``제750조``, ``제3조의2``, ``2018다12345``) and adds Hangul character bigrams so
recall does not depend on morphological analysis. ``KiwiTokenizer`` is an
optional drop-in that uses ``kiwipiepy`` when installed.
"""

from __future__ import annotations

import re
from typing import Protocol

from legal_rag.textutil.refs import CASE_PATTERN

_SPECIAL = re.compile(rf"({CASE_PATTERN}|제\s*\d+\s*조(?:\s*의\s*\d+)?|제\s*\d+\s*항|제\s*\d+\s*호)")
_WORD = re.compile(r"[가-힣]+|[a-z0-9]+")
_HANGUL_LONG = re.compile(r"[가-힣]{3,}")


class Tokenizer(Protocol):
    name: str

    def tokenize(self, text: str) -> list[str]: ...


class RegexTokenizer:
    name = "regex"

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        position = 0
        for match in _SPECIAL.finditer(text):
            tokens += self._words(text[position : match.start()])
            tokens.append(re.sub(r"\s+", "", match.group(0)))
            position = match.end()
        tokens += self._words(text[position:])
        return tokens

    @staticmethod
    def _words(segment: str) -> list[str]:
        out: list[str] = []
        for word in _WORD.findall(segment.lower()):
            out.append(word)
            if _HANGUL_LONG.fullmatch(word):
                out += [word[i : i + 2] for i in range(len(word) - 1)]
        return out


class KiwiTokenizer:
    """Morpheme tokenizer; keeps the same legal identifiers as ``RegexTokenizer``."""

    name = "kiwi"
    _KEEP_TAGS = ("NN", "VV", "VA", "XR", "SL", "SN", "SH")

    def __init__(self) -> None:
        try:
            from kiwipiepy import Kiwi
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("KiwiTokenizer requires the `kiwi` extra: uv sync --extra kiwi") from exc
        self._kiwi = Kiwi()

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        position = 0
        for match in _SPECIAL.finditer(text):
            tokens += self._morphs(text[position : match.start()])
            tokens.append(re.sub(r"\s+", "", match.group(0)))
            position = match.end()
        tokens += self._morphs(text[position:])
        return tokens

    def _morphs(self, segment: str) -> list[str]:
        if not segment.strip():
            return []
        return [token.form.lower() for token in self._kiwi.tokenize(segment) if token.tag.startswith(self._KEEP_TAGS)]


def build_tokenizer(name: str) -> Tokenizer:
    if name == "regex":
        return RegexTokenizer()
    if name == "kiwi":
        return KiwiTokenizer()
    raise ValueError(f"unknown tokenizer {name!r}")

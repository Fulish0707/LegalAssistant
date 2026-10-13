"""Judge-free comparison of an answer with its reference.

The Qwen3:8B answer grader moved 14 of 80 grades between two Gemini runs whose answers differed
little (2026-10-05). These measures use no model: they say how much of the reference's wording,
article numbers and quantities an answer contains. They measure overlap with the reference, not
legal correctness, and recall grows with answer length, so compare answers cut to one length.
"""

from __future__ import annotations

import collections
import random
import re

_ARTICLE = re.compile(r"제\s*\d+\s*조(?:\s*의\s*\d+)?")
_QUANTITY = re.compile(r"\d+(?:\.\d+)?\s*(?:년|개월|일|원|만원|%|퍼센트|명|회|세)")


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _bigrams(text: str) -> collections.Counter:
    squashed = _squash(text)
    return collections.Counter(squashed[i:i + 2] for i in range(len(squashed) - 1))


def bigram_recall(answer: str, reference: str) -> float:
    """Share of the reference's character bigrams (whitespace removed) that the answer contains."""
    wanted, found = _bigrams(reference), _bigrams(answer)
    total = sum(wanted.values())
    return sum(min(count, found[gram]) for gram, count in wanted.items()) / total if total else 0.0


def _facts(text: str) -> set[str]:
    return {_squash(match) for pattern in (_ARTICLE, _QUANTITY) for match in pattern.findall(text or "")}


def fact_recall(answer: str, reference: str) -> float | None:
    """Share of the reference's article numbers and quantities found in the answer; None if it has none."""
    wanted = _facts(reference)
    return len(wanted & _facts(answer)) / len(wanted) if wanted else None


def paired_difference(base: list[float], new: list[float], *, iters: int = 10000, seed: int = 20261005) -> tuple[float, tuple[float, float]]:
    """Mean of new - base over paired items, with a paired bootstrap 95% interval."""
    diffs = [b - a for a, b in zip(base, new, strict=True)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(iters))
    return sum(diffs) / len(diffs), (means[int(0.025 * iters)], means[int(0.975 * iters)])

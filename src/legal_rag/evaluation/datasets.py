"""Read JSONL evaluation datasets."""

from __future__ import annotations

from pathlib import Path

from legal_rag.schemas import GenerationExample, RetrievalExample


def _lines(path: Path) -> list[str]:
    return [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_retrieval_examples(path: Path) -> list[RetrievalExample]:
    return [RetrievalExample.model_validate_json(line) for line in _lines(path)]


def load_generation_examples(path: Path) -> list[GenerationExample]:
    return [GenerationExample.model_validate_json(line) for line in _lines(path)]

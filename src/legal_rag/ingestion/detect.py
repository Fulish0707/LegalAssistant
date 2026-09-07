"""Recognize a raw record's layout by its key fingerprint."""

from __future__ import annotations

from typing import Any

SchemaName = str

_FINGERPRINTS: list[tuple[SchemaName, tuple[str, ...]]] = [
    ("aihub_judgment_v1", ("info", "facts", "dcss", "close")),
    ("statute_json_v1", ("law_name", "articles")),
    ("admin_interpretation_v1", ("authority", "reference_number", "question", "interpretation")),
]


def detect_schema(payload: dict[str, Any]) -> SchemaName | None:
    keys = set(payload)
    for name, required in _FINGERPRINTS:
        if all(key in keys for key in required):
            return name
    return None


def known_schemas() -> list[SchemaName]:
    return [name for name, _ in _FINGERPRINTS]

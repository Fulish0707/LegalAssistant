"""Shared helpers for layout-specific normalizers."""

from __future__ import annotations

from typing import Any


class NormalizationError(ValueError):
    """Raised when a raw record cannot be turned into a canonical document."""


def as_text(value: Any) -> str:
    """AI Hub fields are sometimes strings and sometimes lists of strings."""
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(as_text(item) for item in value if item is not None).strip()
    return str(value).strip()


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if item is not None and str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def require(payload: dict[str, Any], key: str, context: str) -> Any:
    value = payload.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise NormalizationError(f"{context}: missing required field {key!r}")
    return value

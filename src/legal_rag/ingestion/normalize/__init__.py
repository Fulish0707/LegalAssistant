"""Layout-specific normalizers that produce canonical documents."""

from __future__ import annotations

from typing import Any, Callable

from legal_rag.schemas import LegalDocument

from .admin_interpretation import normalize_admin_interpretation
from .aihub_judgment import normalize_aihub_judgment
from .base import NormalizationError
from .statute_json import normalize_statute_json

Normalizer = Callable[..., LegalDocument]

_NORMALIZERS: dict[str, Normalizer] = {
    "aihub_judgment_v1": normalize_aihub_judgment,
    "statute_json_v1": normalize_statute_json,
    "admin_interpretation_v1": normalize_admin_interpretation,
}


def normalize(schema: str, payload: dict[str, Any], *, dataset: str, file: str) -> LegalDocument:
    try:
        normalizer = _NORMALIZERS[schema]
    except KeyError:
        raise NormalizationError(f"unknown schema {schema!r}") from None
    return normalizer(payload, dataset=dataset, file=file)


__all__ = ["NormalizationError", "normalize"]

"""First-seen-wins deduplication on record key, then on normalized text."""

from __future__ import annotations

import hashlib
import re

from legal_rag.schemas import LegalDocument


def text_fingerprint(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(collapsed.encode("utf-8")).hexdigest()


class Deduplicator:
    def __init__(self) -> None:
        self._keys: set[tuple[str, str]] = set()
        self._fingerprints: set[str] = set()

    def is_duplicate(self, doc: LegalDocument) -> str | None:
        key = (doc.source.dataset, doc.source.record_key)
        if key in self._keys:
            return "duplicate record key"
        fingerprint = text_fingerprint(doc.text)
        if fingerprint in self._fingerprints:
            return "duplicate text"
        self._keys.add(key)
        self._fingerprints.add(fingerprint)
        return None

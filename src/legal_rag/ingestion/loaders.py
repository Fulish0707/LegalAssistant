"""Discover raw legal records without interpreting them."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class RawRecord:
    dataset: str  # first path segment under the raw directory
    file: str  # path relative to the raw directory, POSIX separators
    index: int  # position within the file
    payload: dict[str, Any]
    file_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_payloads(path: Path) -> Iterator[Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        for line in text.splitlines():
            if line.strip():
                yield json.loads(line)
        return
    data = json.loads(text)
    if isinstance(data, list):
        yield from data
    else:
        yield data


def discover_raw_records(raw_dir: Path) -> Iterator[RawRecord]:
    """Yield every JSON object under ``raw_dir`` in a stable, sorted order."""
    raw_dir = Path(raw_dir)
    files = sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.suffix in {".json", ".jsonl"})
    for path in files:
        rel = path.relative_to(raw_dir).as_posix()
        dataset = rel.split("/", 1)[0] if "/" in rel else "root"
        digest = sha256_file(path)
        for index, payload in enumerate(_iter_payloads(path)):
            if isinstance(payload, dict):
                yield RawRecord(dataset, rel, index, payload, digest)

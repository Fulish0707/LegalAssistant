"""Provenance for a processed corpus: what went in, what came out, and why records were dropped."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from legal_rag.config import Settings

from .loaders import RawRecord


class RawFileEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str
    size: int
    sha256: str


class DroppedRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str
    index: int
    stage: str  # detect | normalize | validate | dedupe
    reason: str


class Counts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: int = 0
    recognized: int = 0
    unrecognized: int = 0
    normalized: int = 0
    normalization_errors: int = 0
    validation_failures: int = 0
    deduplicated: int = 0
    documents: int = 0
    chunks: int = 0


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corpus_version: str
    pipeline_version: str
    config_hash: str
    created_at: str
    raw_files: list[RawFileEntry] = Field(default_factory=list)
    counts: Counts = Field(default_factory=Counts)
    dropped: list[DroppedRecord] = Field(default_factory=list)

    @classmethod
    def new(cls, settings: Settings, pipeline_version: str) -> "Manifest":
        return cls(
            corpus_version=settings.corpus.version,
            pipeline_version=pipeline_version,
            config_hash=settings.hash(),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    def drop(self, record: RawRecord, stage: str, reason: str) -> None:
        self.dropped.append(DroppedRecord(file=record.file, index=record.index, stage=stage, reason=reason))

    def write(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> "Manifest":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

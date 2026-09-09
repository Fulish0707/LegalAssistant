"""Raw records → validated, deduplicated canonical documents → chunks, reproducibly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel

from legal_rag.chunking import chunk_document
from legal_rag.config import Settings
from legal_rag.schemas import LegalDocument

from . import PIPELINE_VERSION
from .dedupe import Deduplicator
from .detect import detect_schema
from .loaders import discover_raw_records
from .manifest import Manifest, RawFileEntry
from .normalize import NormalizationError, normalize
from .validate import validate


@dataclass(frozen=True)
class IngestionReport:
    documents: int
    chunks: int
    deduplicated: int
    unrecognized: int
    output_dir: Path


def _write_jsonl(path: Path, rows: Iterable[BaseModel]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(row.model_dump_json())
            handle.write("\n")


def run_ingestion(settings: Settings) -> IngestionReport:
    raw_dir = settings.raw_dir()
    out_dir = settings.processed_corpus_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = Manifest.new(settings, PIPELINE_VERSION)
    dedupe = Deduplicator()
    documents: list[LegalDocument] = []
    seen_files: dict[str, RawFileEntry] = {}

    for record in discover_raw_records(raw_dir):
        if record.file not in seen_files:
            size = (raw_dir / record.file).stat().st_size
            seen_files[record.file] = RawFileEntry(file=record.file, size=size, sha256=record.file_sha256)
        manifest.counts.records += 1

        schema = detect_schema(record.payload)
        if schema is None:
            manifest.counts.unrecognized += 1
            manifest.drop(record, "detect", "unrecognized layout")
            continue
        manifest.counts.recognized += 1

        try:
            doc = normalize(schema, record.payload, dataset=record.dataset, file=record.file)
        except NormalizationError as exc:
            manifest.counts.normalization_errors += 1
            manifest.drop(record, "normalize", str(exc))
            continue
        manifest.counts.normalized += 1

        failure = validate(doc)
        if failure is not None:
            manifest.counts.validation_failures += 1
            manifest.drop(record, "validate", failure.reason)
            continue

        reason = dedupe.is_duplicate(doc)
        if reason is not None:
            manifest.counts.deduplicated += 1
            manifest.drop(record, "dedupe", reason)
            continue
        documents.append(doc)

    documents.sort(key=lambda doc: doc.id)
    chunks = [chunk for doc in documents for chunk in chunk_document(doc, corpus_version=settings.corpus.version)]

    _write_jsonl(out_dir / "documents.jsonl", documents)
    _write_jsonl(out_dir / "chunks.jsonl", chunks)
    manifest.raw_files = [seen_files[name] for name in sorted(seen_files)]
    manifest.counts.documents = len(documents)
    manifest.counts.chunks = len(chunks)
    manifest.write(out_dir / "manifest.json")

    return IngestionReport(
        documents=len(documents),
        chunks=len(chunks),
        deduplicated=manifest.counts.deduplicated,
        unrecognized=manifest.counts.unrecognized,
        output_dir=out_dir,
    )

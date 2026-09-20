"""Re-derive a corpus's chunks from its own documents, structurally.

Documents are copied byte for byte — the new version differs only in how the text was
cut. Output is staged in a temporary directory and renamed into place, so a failure
leaves no half-written corpus and an existing destination is never overwritten.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

from legal_rag.chunking.imported import ChunkingOptions, chunk_imported_document
from legal_rag.schemas.documents import ImportedDocument

PIPELINE_VERSION = "structure_rechunk_v1"


def _percentiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {"count": 0, "min": 0, "median": 0, "p90": 0, "p99": 0, "max": 0}
    ordered = sorted(values)

    def at(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": at(0.5),
        "p90": at(0.9),
        "p99": at(0.99),
        "max": ordered[-1],
    }


def rechunk_corpus(
    source: Path,
    destination: Path,
    *,
    corpus_version: str,
    options: ChunkingOptions | None = None,
    limit: int | None = None,
    examples: int = 3,
) -> dict:
    """Write ``destination`` with the documents of ``source`` and structurally re-cut chunks."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f"{destination} already exists; choose a new output version")
    documents = source / "documents.jsonl"
    if not documents.exists():
        raise FileNotFoundError(f"no processed corpus at {source}")
    options = options or ChunkingOptions()

    parsed: Counter = Counter()
    fallback: Counter = Counter()
    kinds: Counter = Counter()
    per_type_chunks: Counter = Counter()
    per_type_lengths: dict[str, list[int]] = {}
    samples: dict[str, list[str]] = {}
    lengths: list[int] = []
    structural = 0
    total_documents = 0
    total_chunks = 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rechunk-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "corpus"
        staging.mkdir()
        with (
            documents.open(encoding="utf-8") as source_handle,
            (staging / "documents.jsonl").open("w", encoding="utf-8") as document_out,
            (staging / "chunks.jsonl").open("w", encoding="utf-8") as chunk_out,
        ):
            for number, line in enumerate(source_handle, 1):
                if not line.strip():
                    continue
                if limit is not None and total_documents >= limit:
                    break
                try:
                    document = ImportedDocument.model_validate_json(line)
                except Exception as exc:  # noqa: BLE001 - one bad line names itself
                    raise ValueError(f"{documents}:{number}: {exc}") from exc
                document_out.write(line if line.endswith("\n") else line + "\n")
                total_documents += 1
                kind = document.document_type.value
                chunks, report = chunk_imported_document(document, corpus_version, options)
                if report.parsed:
                    parsed[kind] += 1
                    kinds[report.kind or "?"] += 1
                else:
                    fallback[kind] += 1
                for chunk in chunks:
                    chunk_out.write(chunk.model_dump_json() + "\n")
                    lengths.append(len(chunk.text))
                    per_type_lengths.setdefault(kind, []).append(len(chunk.text))
                    per_type_chunks[kind] += 1
                    total_chunks += 1
                    if report.parsed:
                        structural += 1
                    bucket = samples.setdefault(kind, [])
                    if len(bucket) < examples:
                        bucket.append(chunk.text[:200])

        if not total_chunks:
            raise ValueError("no chunks produced")
        report = {
            "corpus_version": corpus_version,
            "pipeline_version": PIPELINE_VERSION,
            "source": str(source.resolve()),
            "options": options.__dict__,
            "documents": total_documents,
            "chunks": total_chunks,
            "chunks_per_document": round(total_chunks / total_documents, 3),
            "parsed": dict(parsed),
            "fallback": dict(fallback),
            "parser_kinds": dict(kinds),
            "chunks_by_type": dict(per_type_chunks),
            "chunk_chars": _percentiles(lengths),
            "chunk_chars_by_type": {kind: _percentiles(values) for kind, values in per_type_lengths.items()},
            # Every chunk opens with a header line by construction; the second share is the
            # one that matters — chunks whose header is a real section path, not a window label.
            "header_line_share": 1.0,
            "structural_heading_share": round(structural / total_chunks, 4),
            "examples": samples,
        }
        manifest = {
            "corpus_version": corpus_version,
            "pipeline_version": PIPELINE_VERSION,
            "config_hash": "",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "raw_files": [],
            "counts": {"documents": total_documents, "chunks": total_chunks},
            "dropped": [],
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (staging / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if destination.exists():
            raise FileExistsError(f"{destination} appeared during rechunking")
        shutil.move(str(staging), str(destination))
    return report


__all__ = ["rechunk_corpus", "PIPELINE_VERSION"]

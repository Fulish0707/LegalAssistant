"""Validate and adapt external documents/chunks without reading QA or changing sources."""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from datetime import date
import hashlib
import json
from pathlib import Path
import tempfile

from legal_rag.config import Settings
from legal_rag.schemas import Chunk, ChunkMetadata, CitationKind, DocumentType, SourceRef, citation_kind
from legal_rag.schemas.documents import ImportedDocument
from .manifest import Manifest, RawFileEntry


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _rows(path: Path):
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line, object_pairs_hook=_unique_object)
                if not isinstance(record, dict):
                    raise ValueError("expected JSON object")
                yield number, record
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc


def _text(record: dict, key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty string")
    return value


def _header(record: dict) -> dict:
    return {key: value for key, value in record.items() if key != "text"}


def _date(value) -> date | None:
    # Dates are advisory metadata: preserve unparseable originals rather than inventing dates.
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10].replace('.', '-'))
    except ValueError:
        return None


def _document(record: dict, dataset: str, source_file: str) -> ImportedDocument:
    identifier = _text(record, 'document_id')
    return ImportedDocument(
        id=identifier, document_type=DocumentType(_text(record, 'document_type')),
        legal_domain=_text(record, 'legal_domain'), title=record.get('title') or '',
        text=_text(record, 'text'), source=SourceRef(dataset=dataset, file=source_file, record_key=identifier),
        metadata={'original': _header(record)},
    )


def _chunk(record: dict, dataset: str, version: str) -> Chunk:
    position = record.get('chunk_index')
    total = record.get('total_chunks')
    if type(position) is not int or type(total) is not int or not 0 <= position < total:
        raise ValueError('chunk_index and total_chunks must be integers with 0 <= chunk_index < total_chunks')
    kind = DocumentType(_text(record, 'document_type'))
    original_metadata = record.get('metadata') or {}
    if not isinstance(original_metadata, dict):
        raise ValueError('metadata must be an object')
    return Chunk(
        id=_text(record, 'chunk_id'), document_id=_text(record, 'document_id'),
        document_type=kind, path=f'chunk:{position:04d}', order=position,
        text=_text(record, 'text'), title=record.get('title') or '',
        metadata=ChunkMetadata(
            dataset=dataset, corpus_version=version, legal_domain=_text(record, 'legal_domain'),
            authority=record.get('authority'),
            court=record.get('authority') if citation_kind(kind) is CitationKind.case else None,
            case_number=original_metadata.get('case_number'),
            law_name=record.get('title') if kind == DocumentType.statute else None,
            decision_date=_date(original_metadata.get('decision_date')),
            effective_date=_date(original_metadata.get('effective_date')),
            original=_header(record),
        ),
    )


def import_processed(input_dir: Path, settings: Settings, *, max_documents: int | None = None,
                     validate_only: bool = False) -> dict:
    """Validate the entire input and atomically publish a new canonical corpus.

    A sample includes every chunk of the first N documents. Validation still covers
    all input records. Existing destination directories are never overwritten.
    """
    if max_documents is not None and max_documents < 1:
        raise ValueError('max_documents must be positive')
    input_dir = Path(input_dir).resolve()
    destination = settings.processed_corpus_dir().resolve()
    if not validate_only and destination.exists():
        raise FileExistsError(f'{destination} already exists; choose a new corpus version')
    paths = [input_dir / name for name in ('documents.jsonl', 'chunks.jsonl')]
    fingerprints = []
    for path in paths:
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        fingerprints.append(RawFileEntry(file=str(path), size=path.stat().st_size, sha256=digest))
    dataset = 'aihub_preprocessed'
    docs: dict[str, tuple[str, str]] = {}
    selected: set[str] = set()
    chunk_ids: set[str] = set()
    positions: dict[str, set[int]] = {}
    declared: dict[str, int] = {}
    types: Counter = Counter()
    counts = {'documents': 0, 'chunks': 0, 'source_documents': 0, 'source_chunks': 0}
    if not validate_only:
        destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.import-', dir=None if validate_only else destination.parent) as temporary:
        staging = Path(temporary) / 'corpus'
        staging.mkdir()
        with ExitStack() as stack:
            doc_out = stack.enter_context((staging / 'documents.jsonl').open('w', encoding='utf-8')) if not validate_only else None
            chunk_out = stack.enter_context((staging / 'chunks.jsonl').open('w', encoding='utf-8')) if not validate_only else None
            for number, record in _rows(paths[0]):
                try:
                    document = _document(record, dataset, str(paths[0]))
                    if document.id in docs:
                        raise ValueError(f'duplicate document ID {document.id!r}')
                    docs[document.id] = (document.document_type.value, document.legal_domain)
                    counts['source_documents'] += 1
                    if max_documents is None or len(selected) < max_documents:
                        selected.add(document.id)
                        counts['documents'] += 1
                        if doc_out:
                            doc_out.write(document.model_dump_json() + '\n')
                except (ValueError, TypeError) as exc:
                    raise ValueError(f'{paths[0]}:{number}: {exc}') from exc
            for number, record in _rows(paths[1]):
                try:
                    chunk = _chunk(record, dataset, settings.corpus.version)
                    if chunk.id in chunk_ids:
                        raise ValueError(f'duplicate chunk ID {chunk.id!r}')
                    chunk_ids.add(chunk.id)
                    if chunk.document_id not in docs:
                        raise ValueError(f'missing document {chunk.document_id!r}')
                    if docs[chunk.document_id] != (chunk.document_type.value, chunk.metadata.legal_domain):
                        raise ValueError('chunk domain/type differs from its document')
                    seen = positions.setdefault(chunk.document_id, set())
                    if chunk.order in seen:
                        raise ValueError('duplicate chunk position for document')
                    seen.add(chunk.order)
                    total = record['total_chunks']
                    if declared.setdefault(chunk.document_id, total) != total:
                        raise ValueError('inconsistent total_chunks for document')
                    counts['source_chunks'] += 1
                    types[chunk.document_type.value] += 1
                    if chunk.document_id in selected:
                        counts['chunks'] += 1
                        if chunk_out:
                            chunk_out.write(chunk.model_dump_json() + '\n')
                except (ValueError, TypeError) as exc:
                    raise ValueError(f'{paths[1]}:{number}: {exc}') from exc
        if not docs or not chunk_ids:
            raise ValueError('documents and chunks must both be nonempty')
        if not counts['chunks']:
            raise ValueError('selected documents have no chunks')
        documents_without_chunks = [identifier for identifier in docs if identifier not in positions]
        for identifier in positions:
            if len(positions[identifier]) != declared[identifier]:
                raise ValueError(f'incomplete chunks for document {identifier!r}')
        # Ensure the provenance hashes describe exactly the source snapshot imported.
        for path, fingerprint in zip(paths, fingerprints):
            with path.open('rb') as handle:
                if hashlib.file_digest(handle, 'sha256').hexdigest() != fingerprint.sha256:
                    raise ValueError(f'source changed during import: {path}')
        report = dict(counts, status='valid', document_types=dict(types),
                      documents_without_chunks=documents_without_chunks,
                      source_files=[f.model_dump() for f in fingerprints],
                      qa_leakage_check='not performed; QA files are never read by this importer')
        if not validate_only:
            manifest = Manifest.new(settings, 'external_processed_v1')
            manifest.raw_files = fingerprints
            manifest.counts.documents = counts['documents']
            manifest.counts.chunks = counts['chunks']
            manifest.write(staging / 'manifest.json')
            (staging / 'validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            if destination.exists():
                raise FileExistsError(f'{destination} appeared during import')
            staging.rename(destination)
    return report

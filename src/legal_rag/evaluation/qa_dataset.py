"""Turn an external ``qa.jsonl`` into a retrieval dataset.

Only the question and its document reference cross over. The gold answer is training
and grading data; it never becomes a query, a filter, or anything the retriever sees.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from legal_rag.schemas import DocumentType, RetrievalExample, RetrievalFilters


@dataclass
class QaDatasetReport:
    total: int = 0
    kept: int = 0
    dropped_invalid: int = 0
    dropped_unmatched: int = 0
    dropped_duplicate: int = 0
    unmatched_sample: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "total": self.total,
            "kept": self.kept,
            "dropped_invalid": self.dropped_invalid,
            "dropped_unmatched": self.dropped_unmatched,
            "dropped_duplicate": self.dropped_duplicate,
            "unmatched_sample": self.unmatched_sample,
        }


_UNMATCHED_SAMPLE = 10


def _rows(path: Path) -> Iterable[tuple[int, dict]]:
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("expected JSON object")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path.name}:{number}: {exc}") from exc
            yield number, record


def _text(record: dict, key: str) -> str | None:
    value = record.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def example_id(document_id: str, question: str) -> str:
    """Stable across runs and independent of row order, so reports stay comparable."""
    digest = hashlib.sha256(f"{document_id}|{question}".encode("utf-8")).hexdigest()[:16]
    return f"qa:{digest}"


def _filters(record: dict) -> RetrievalFilters | None:
    domain = _text(record, "legal_domain")
    raw_type = _text(record, "document_type")
    try:
        types = [DocumentType(raw_type)] if raw_type else None
    except ValueError:
        types = None
    if not domain and not types:
        return None
    return RetrievalFilters(legal_domain=domain, document_types=types)


def corpus_document_ids(corpus_dir: Path) -> set[str]:
    """Read only the ``id`` of each document; full validation would parse gigabytes."""
    path = Path(corpus_dir) / "documents.jsonl"
    identifiers: set[str] = set()
    for number, record in _rows(path):
        identifier = _text(record, "id")
        if not identifier:
            raise ValueError(f"{path.name}:{number}: document has no id")
        identifiers.add(identifier)
    return identifiers


def build_retrieval_examples(
    qa_path: Path,
    known_document_ids: set[str],
    *,
    with_filters: bool = False,
    max_examples: int | None = None,
    seed: int = 0,
) -> tuple[list[RetrievalExample], QaDatasetReport]:
    """Read ``qa.jsonl`` and keep the questions whose document is actually in the corpus.

    A question whose document never matched a source cannot be retrieved, so keeping it
    would depress Recall@K for a data-matching reason rather than a retrieval one.
    """
    qa_path = Path(qa_path)
    report = QaDatasetReport()
    by_id: dict[str, RetrievalExample] = {}
    for _, record in _rows(qa_path):
        report.total += 1
        question = _text(record, "question")
        document_id = _text(record, "document_id")
        if not question or not document_id:
            report.dropped_invalid += 1
            continue
        if document_id not in known_document_ids:
            report.dropped_unmatched += 1
            if len(report.unmatched_sample) < _UNMATCHED_SAMPLE:
                report.unmatched_sample.append(document_id)
            continue
        identifier = example_id(document_id, question)
        if identifier in by_id:
            report.dropped_duplicate += 1
            continue
        by_id[identifier] = RetrievalExample(
            id=identifier,
            query=question,
            relevant_document_ids=[document_id],
            filters=_filters(record) if with_filters else None,
        )
    examples = [by_id[key] for key in sorted(by_id)]
    if max_examples is not None and len(examples) > max_examples:
        examples = sorted(random.Random(seed).sample(examples, max_examples), key=lambda e: e.id)
    report.kept = len(examples)
    if not examples:
        raise ValueError(f"{qa_path}: no usable questions; {report.summary()}")
    return examples, report


def write_retrieval_examples(examples: Iterable[RetrievalExample], path: Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(example.model_dump_json(exclude_none=True) + "\n")
            written += 1
    return written

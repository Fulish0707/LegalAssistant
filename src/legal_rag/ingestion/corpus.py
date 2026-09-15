"""Read access to a processed corpus directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from pydantic import TypeAdapter

from legal_rag.schemas import Chunk, LegalDocument

from .manifest import Manifest
from legal_rag.schemas.documents import ImportedDocument

_DOCUMENT_ADAPTER: TypeAdapter = TypeAdapter(LegalDocument)


class Corpus:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._documents_by_id: dict[str, LegalDocument | ImportedDocument] | None = None

    @classmethod
    def load(cls, directory: Path) -> "Corpus":
        directory = Path(directory)
        if not (directory / "documents.jsonl").exists():
            raise FileNotFoundError(f"no processed corpus at {directory}")
        return cls(directory)

    def documents(self) -> Iterator[LegalDocument | ImportedDocument]:
        with (self.directory / "documents.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if record.get("format") == "external_processed_v1":
                        yield ImportedDocument.model_validate(record)
                    else:
                        yield _DOCUMENT_ADAPTER.validate_python(record)

    def chunks(self) -> Iterator[Chunk]:
        with (self.directory / "chunks.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield Chunk.model_validate_json(line)

    def document(self, document_id: str) -> LegalDocument | ImportedDocument:
        if self._documents_by_id is None:
            self._documents_by_id = {doc.id: doc for doc in self.documents()}
        return self._documents_by_id[document_id]

    def manifest(self) -> Manifest:
        return Manifest.read(self.directory / "manifest.json")

    def counts(self) -> dict[str, int]:
        return json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))["counts"]

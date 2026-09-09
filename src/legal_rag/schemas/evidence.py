"""Evidence handed to the generator and the verifier."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .documents import DocumentType


class Evidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str  # "S1", "S2", ... stable within one turn
    chunk_id: str
    document_id: str
    document_type: DocumentType
    header: dict[str, str]
    text: str
    score: float


class EvidenceBundle(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    items: list[Evidence] = Field(default_factory=list)

    def by_id(self) -> dict[str, Evidence]:
        return {item.source_id: item for item in self.items}

    def render(self) -> str:
        blocks = []
        for item in self.items:
            lines = [f"[LEGAL SOURCE {item.source_id}]", ""]
            lines += [f"{key}: {value}" for key, value in item.header.items()]
            lines += [f"Source ID: {item.source_id}", "", "Text:", item.text, ""]
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

"""Evidence handed to the generator and the verifier."""

from __future__ import annotations

from collections.abc import Sequence

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


class UserDocument(BaseModel):
    """A passage of a file the user uploaded: a fact of their case, never legal authority."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=4000)


def render_user_documents(documents: Sequence[UserDocument]) -> str:
    return "\n\n".join(f"[USER DOCUMENT D{number}]\nFile: {document.name}\n\nText:\n{document.text}"
                       for number, document in enumerate(documents, 1))

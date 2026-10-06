"""Evidence handed to the generator and the verifier."""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from .documents import DocumentType


# Calendar dates and money amounts: the facts of one case. Periods ("2년"), rates and deadlines stay.
_CASE_DATE = re.compile(r"\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.?|\d{4}년\s*\d{1,2}월\s*\d{1,2}일")
_CASE_AMOUNT = re.compile(r"\d[\d,]*\s*(?:억\s*)?(?:\d[\d,]*\s*)?(?:천만|백만|만)?\s*원")


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

    def render(self, *, mask_case_figures: bool = False) -> str:
        """The evidence as the model reads it. With `mask_case_figures`, the dates and amounts in
        a judgment's or an interpretation's text are hidden: they belong to that case, not to
        the user's. Statutes are shown whole, and the response keeps every text unmasked."""
        blocks = []
        for item in self.items:
            text = item.text
            if mask_case_figures and item.header.get("Type") != "Statute":
                text = _CASE_AMOUNT.sub("(금액)", _CASE_DATE.sub("(날짜)", text))
            lines = [f"[LEGAL SOURCE {item.source_id}]", ""]
            lines += [f"{key}: {value}" for key, value in item.header.items()]
            lines += [f"Source ID: {item.source_id}", "", "Text:", text, ""]
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

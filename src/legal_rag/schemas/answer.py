"""The structured answer contract the LLM must return."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CitationKind(StrEnum):
    statute = "statute"
    case = "case"
    interpretation = "interpretation"


class Confidence(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    kind: CitationKind
    reference: str  # "민법 제750조" or a case number as written by the model
    quote: str | None = None


class LegalAnswer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    summary: str
    legal_issues: list[str] = Field(default_factory=list)
    analysis: str
    possible_actions: list[str] = Field(default_factory=list)
    additional_information_needed: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    uncertainty: str = ""
    confidence: Confidence

"""Explicit scope and bounded non-answer details for the fictional lease pilot."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PolicyScope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    profile: Literal["lease_review_v1"] = "lease_review_v1"
    matter_id: str = Field(min_length=1)
    issue_id: Literal["rent-deduction", "clause-explanation"]


class WithheldDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason_code: str
    known_facts: list[str] = Field(default_factory=list)
    unresolved_items: list[str] = Field(default_factory=list)
    explanation: str

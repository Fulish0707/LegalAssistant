"""What a chat turn returns to callers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .answer import LegalAnswer
from .evidence import EvidenceBundle
from .trace import Trace
from .verification import VerificationReport


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["answer", "clarification", "rejected"]
    conversation_id: str
    answer: LegalAnswer | None = None
    questions: list[str] = Field(default_factory=list)
    reason: str | None = None
    evidence: EvidenceBundle | None = None
    verification: VerificationReport | None = None
    trace: Trace | None = None

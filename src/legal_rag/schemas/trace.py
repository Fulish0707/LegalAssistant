"""Per-turn trace for observability and audit. Mutable: built incrementally."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .query import QueryAnalysis
from .verification import VerificationReport


class StageTiming(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str
    ms: float


class Trace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    turn_id: str
    model: str | None = None
    timings: list[StageTiming] = Field(default_factory=list)
    analysis: QueryAnalysis | None = None
    search_queries: list[str] = Field(default_factory=list)
    retrieved: dict[str, list[tuple[str, float]]] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    verification: VerificationReport | None = None
    policy_action: str | None = None
    regenerations: int = 0
    user_text: str | None = None

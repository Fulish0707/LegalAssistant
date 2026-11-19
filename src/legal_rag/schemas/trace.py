"""Per-turn trace for observability and audit. Mutable: built incrementally."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .query import QueryAnalysis
from .verification import VerificationReport


class StageTiming(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str
    ms: float


class PostRetrievalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str
    reason: str
    bundle_revision: int
    evidence_ids: list[str] = Field(default_factory=list)
    # Source id -> role at each cited case/interpretation quote (quote_roles); "unknown" is allowed.
    support_roles: dict[str, str] = Field(default_factory=dict)


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
    post_retrieval: list[PostRetrievalDecision] = Field(default_factory=list)
    # Case/interpretation sources the pilot answer cites whose quoted role is not confirmed
    # reasoning or a legal rule; the answer then carries a fixed notice.
    unverified_case_sources: list[str] = Field(default_factory=list)
    user_text: str | None = None

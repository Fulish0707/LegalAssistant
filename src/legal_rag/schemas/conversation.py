"""Conversation state, kept apart from legal evidence."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Turn(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    evidence_ids: list[str] = Field(default_factory=list)  # chunk ids, audit only


class Conversation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    turns: list[Turn] = Field(default_factory=list)

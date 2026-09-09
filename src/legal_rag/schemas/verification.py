"""Verification results and policy actions."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

IssueKind = Literal[
    "unknown_source_id",
    "source_id_mismatch",
    "statute_not_in_evidence",
    "case_not_in_evidence",
]


class VerificationIssue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: IssueKind
    detail: str


class VerificationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ok: bool
    issues: list[VerificationIssue] = Field(default_factory=list)
    checked_statutes: list[str] = Field(default_factory=list)
    checked_cases: list[str] = Field(default_factory=list)


class VerificationAction(StrEnum):
    accept = "accept"
    regenerate = "regenerate"
    strip = "strip"
    reject = "reject"

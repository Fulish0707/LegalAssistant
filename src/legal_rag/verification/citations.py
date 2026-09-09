"""Reference extraction for verification. Thin wrapper over the leaf text utilities."""

from __future__ import annotations

from legal_rag.schemas import LegalAnswer
from legal_rag.textutil.refs import StatuteRef, extract_case_refs, extract_statute_refs

__all__ = ["StatuteRef", "answer_text", "article_only", "extract_case_refs", "extract_statute_refs"]


def answer_text(answer: LegalAnswer) -> str:
    """Every free-text field in which the model could assert a legal authority."""
    return "\n".join([answer.summary, answer.analysis, *answer.possible_actions, answer.uncertainty])


def article_only(reference: str) -> str:
    refs = extract_statute_refs(reference)
    return refs[0].article if refs else reference

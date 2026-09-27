"""Reference extraction for verification. Thin wrapper over the leaf text utilities."""

from __future__ import annotations

from legal_rag.schemas import LegalAnswer
from legal_rag.textutil.refs import StatuteRef, extract_case_refs, extract_statute_refs

__all__ = ["StatuteRef", "answer_text", "article_only", "extract_case_refs", "extract_statute_refs"]


def answer_text(answer: LegalAnswer) -> str:
    """Every free-text field in which the model could assert a legal authority."""
    # legal_issues and additional_information_needed were once skipped, and "헌법 제11조" in a
    # follow-up list passed verification (2026-09-27).
    return "\n".join([answer.summary, *answer.legal_issues, answer.analysis, *answer.possible_actions,
                      *answer.additional_information_needed, answer.uncertainty])


def article_only(reference: str) -> str:
    refs = extract_statute_refs(reference)
    return refs[0].article if refs else reference

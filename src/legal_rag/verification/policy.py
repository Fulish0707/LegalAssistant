"""What to do with a verification report: accept, regenerate once, strip, or reject."""

from __future__ import annotations

from legal_rag.schemas import Confidence, LegalAnswer, VerificationAction, VerificationReport
from legal_rag.textutil.sentences import split_sentences

from .citations import article_only

_STRIP_NOTE = "다음 인용은 제공된 근거에서 확인되지 않아 제거되었습니다: "


class VerificationPolicy:
    def __init__(self, max_regenerations: int = 1) -> None:
        self.max_regenerations = max_regenerations

    def decide(self, report: VerificationReport, attempt: int) -> VerificationAction:
        if report.ok:
            return VerificationAction.accept
        return VerificationAction.regenerate if attempt < self.max_regenerations else VerificationAction.strip

    def feedback(self, report: VerificationReport) -> str:
        listed = "; ".join(f"{issue.kind}: {issue.detail}" for issue in report.issues)
        return f"이전 답변에서 근거 없는 인용이 발견되었습니다: {listed}. 제공된 [LEGAL SOURCE] 자료만 인용하고, 근거에 없는 조문과 사건번호는 언급하지 말고 다시 작성하세요."

    def strip(self, answer: LegalAnswer, report: VerificationReport) -> LegalAnswer:
        bad_refs: set[str] = set()
        bad_ids: set[str] = set()
        for issue in report.issues:
            if issue.kind in ("unknown_source_id", "source_id_mismatch"):
                source_id, _, reference = issue.detail.partition(": ")
                bad_refs.add(reference)
                if issue.kind == "unknown_source_id":
                    bad_ids.add(source_id)
            else:
                bad_refs.add(issue.detail)
        needles = {needle for ref in bad_refs for needle in (ref, article_only(ref)) if needle}

        def tainted(text: str) -> bool:
            return any(needle in text for needle in needles)

        def clean(text: str) -> str:
            return " ".join(sentence for sentence in split_sentences(text) if not tainted(sentence))

        citations = [c for c in answer.citations if c.source_id not in bad_ids and not tainted(c.reference)]
        note = _STRIP_NOTE + ", ".join(sorted(bad_refs))
        return answer.model_copy(
            update={
                "citations": citations,
                "summary": clean(answer.summary),
                "analysis": clean(answer.analysis),
                "possible_actions": [action for action in answer.possible_actions if not tainted(action)],
                "legal_issues": [issue for issue in answer.legal_issues if not tainted(issue)],
                "additional_information_needed": [item for item in answer.additional_information_needed if not tainted(item)],
                "confidence": Confidence.low,
                "uncertainty": f"{answer.uncertainty} {note}".strip(),
            }
        )

    @staticmethod
    def decide_after_strip(answer: LegalAnswer, *, evidence_required: bool) -> VerificationAction:
        if evidence_required and not answer.citations and not answer.analysis.strip():
            return VerificationAction.reject
        return VerificationAction.strip

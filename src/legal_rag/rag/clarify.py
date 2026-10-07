"""Decide whether to ask for facts instead of retrieving.

Asking replaces the answer only when there is no legal area to search ("법적으로 어떻게 되나요?").
With an area, the turn is answered from what is known and the open questions ride in the answer,
so each reply narrows it. Questioning first stalled conversations: 8 of 10 replies to a
clarification were questioned again, and 4 of 8 a third time (2026-10-05).
"""

from legal_rag.schemas import QueryAnalysis


class ClarificationPolicy:
    def __init__(self, allow: bool = True) -> None:
        self.allow = allow

    def should_clarify(self, analysis: QueryAnalysis) -> bool:
        if not self.allow or not analysis.needs_more_info or not analysis.missing_facts:
            return False
        # A user who names a specific article or case wants it looked up, not more questions.
        return analysis.domain is None and not (analysis.statute_refs or analysis.case_refs)

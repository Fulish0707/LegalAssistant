"""Decide whether to ask for facts instead of retrieving."""

from legal_rag.schemas import QueryAnalysis


class ClarificationPolicy:
    def __init__(self, allow: bool = True) -> None:
        self.allow = allow

    def should_clarify(self, analysis: QueryAnalysis) -> bool:
        if not self.allow or not analysis.needs_more_info or not analysis.missing_facts:
            return False
        # A user who names a specific article or case wants it looked up, not more questions.
        return not (analysis.statute_refs or analysis.case_refs)

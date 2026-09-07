"""Deterministic grounding checks: every cited authority must exist in the evidence bundle."""

from __future__ import annotations

from legal_rag.schemas import Evidence, EvidenceBundle, LegalAnswer, VerificationIssue, VerificationReport

from .citations import answer_text, extract_case_refs, extract_statute_refs


class Verifier:
    def verify(self, answer: LegalAnswer, bundle: EvidenceBundle) -> VerificationReport:
        by_id = bundle.by_id()
        evidence_statutes: set[str] = set()
        evidence_cases: set[str] = set()
        for item in bundle.items:
            law, article = item.header.get("Law"), item.header.get("Article")
            if article:
                evidence_statutes.add(article)
                if law:
                    evidence_statutes.add(f"{law} {article}")
            if case := item.header.get("Case number"):
                evidence_cases.add(case)
            for ref in extract_statute_refs(item.text):
                evidence_statutes.update({ref.render(), ref.article})
            evidence_cases.update(extract_case_refs(item.text))

        issues: list[VerificationIssue] = []
        for citation in answer.citations:
            evidence = by_id.get(citation.source_id)
            if evidence is None:
                issues.append(VerificationIssue(kind="unknown_source_id", detail=f"{citation.source_id}: {citation.reference}"))
                continue
            if not self._block_supports(evidence, citation.reference):
                issues.append(VerificationIssue(kind="source_id_mismatch", detail=f"{citation.source_id}: {citation.reference}"))

        text = answer_text(answer)
        statute_refs = extract_statute_refs(text)
        for ref in statute_refs:
            if ref.render() not in evidence_statutes and ref.article not in evidence_statutes:
                issues.append(VerificationIssue(kind="statute_not_in_evidence", detail=ref.render()))
        case_refs = extract_case_refs(text)
        for case in case_refs:
            if case not in evidence_cases:
                issues.append(VerificationIssue(kind="case_not_in_evidence", detail=case))

        return VerificationReport(
            ok=not issues,
            issues=issues,
            checked_statutes=[ref.render() for ref in statute_refs],
            checked_cases=case_refs,
        )

    @staticmethod
    def _block_supports(evidence: Evidence, reference: str) -> bool:
        haystack = " ".join([*evidence.header.values(), evidence.text])
        statute_refs = extract_statute_refs(reference)
        if statute_refs:
            ref = statute_refs[0]
            header_article = evidence.header.get("Article")
            header_law = evidence.header.get("Law")
            in_header = header_article == ref.article and (ref.law_name is None or header_law == ref.law_name)
            in_text = any(found.article == ref.article and (ref.law_name is None or found.law_name in (None, ref.law_name)) for found in extract_statute_refs(haystack))
            return in_header or in_text
        cases = extract_case_refs(reference)
        if cases:
            return cases[0] in extract_case_refs(haystack) or evidence.header.get("Case number") == cases[0]
        return reference.strip() in haystack

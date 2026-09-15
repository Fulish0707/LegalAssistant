"""Deterministic citation backfill.

Small models often name the right article in prose but leave ``citations`` empty. When
an authority named in the answer text is present in an evidence block, add a citation
pointing at that block so the answer stays traceable. Nothing is invented: only
references already grounded in the bundle are added.
"""

from __future__ import annotations

from legal_rag.schemas import Citation, CitationKind, Evidence, EvidenceBundle, LegalAnswer, citation_kind

from .citations import answer_text, extract_case_refs, extract_statute_refs


def _kind(evidence: Evidence) -> CitationKind:
    return citation_kind(evidence.document_type)


def _statute_block(bundle: EvidenceBundle, law_name: str | None, article: str) -> Evidence | None:
    for item in bundle.items:
        if item.header.get("Article") == article and (law_name is None or item.header.get("Law") == law_name):
            return item
    for item in bundle.items:
        for ref in extract_statute_refs(item.text):
            if ref.article == article and (law_name is None or ref.law_name in (None, law_name)):
                return item
    return None


def _case_block(bundle: EvidenceBundle, case_number: str) -> Evidence | None:
    for item in bundle.items:
        if item.header.get("Case number") == case_number:
            return item
    for item in bundle.items:
        if case_number in extract_case_refs(item.text):
            return item
    return None


def backfill_citations(answer: LegalAnswer, bundle: EvidenceBundle) -> LegalAnswer:
    existing = {(c.source_id, c.reference) for c in answer.citations}
    cited_refs = {c.reference for c in answer.citations}
    added: list[Citation] = []
    text = answer_text(answer)
    for ref in extract_statute_refs(text):
        rendered = ref.render()
        if rendered in cited_refs or ref.article in cited_refs:
            continue
        block = _statute_block(bundle, ref.law_name, ref.article)
        if block is not None and (block.source_id, rendered) not in existing:
            added.append(Citation(source_id=block.source_id, kind=_kind(block), reference=rendered))
            existing.add((block.source_id, rendered))
            cited_refs.add(rendered)
    for case in extract_case_refs(text):
        if case in cited_refs:
            continue
        block = _case_block(bundle, case)
        if block is not None and (block.source_id, case) not in existing:
            added.append(Citation(source_id=block.source_id, kind=_kind(block), reference=case))
            existing.add((block.source_id, case))
            cited_refs.add(case)
    if not added:
        return answer
    return answer.model_copy(update={"citations": [*answer.citations, *added]})

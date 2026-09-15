"""Turn ranked chunks into numbered evidence blocks with stable source ids."""

from __future__ import annotations

from legal_rag.schemas import CitationKind, Evidence, EvidenceBundle, ScoredChunk, citation_kind


class ContextBuilder:
    def __init__(self, max_chars: int = 1500) -> None:
        self.max_chars = max_chars

    def build(self, hits: list[ScoredChunk]) -> EvidenceBundle:
        items: list[Evidence] = []
        for n, hit in enumerate(hits, start=1):
            chunk = hit.chunk
            m = chunk.metadata
            kind = citation_kind(chunk.document_type)
            if kind is CitationKind.statute:
                header = {
                    "Type": "Statute",
                    "Law": m.law_name or chunk.title,
                    "Article": m.article or "",
                    "Paragraph": m.paragraph or "",
                    "Effective date": m.effective_date.isoformat() if m.effective_date else "",
                }
            elif kind is CitationKind.case:
                header = {
                    "Type": "Case",
                    "Court": m.court or "",
                    "Case number": m.case_number or "",
                    "Decision date": m.decision_date.isoformat() if m.decision_date else "",
                    "Section": m.section or "",
                }
            else:
                header = {"Type": "Interpretation", "Authority": m.authority or "", "Reference number": m.reference_number or ""}
            header = {key: value for key, value in header.items() if value}
            text = chunk.text if len(chunk.text) <= self.max_chars else chunk.text[: self.max_chars] + "…"
            items.append(
                Evidence(
                    source_id=f"S{n}",
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    document_type=chunk.document_type,
                    header=header,
                    text=text,
                    score=hit.score,
                )
            )
        return EvidenceBundle(items=items)

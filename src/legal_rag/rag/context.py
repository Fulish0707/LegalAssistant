"""Turn ranked chunks into numbered evidence blocks with stable source ids."""

from __future__ import annotations

from collections.abc import Callable

from legal_rag.schemas import Chunk, CitationKind, Evidence, EvidenceBundle, ScoredChunk, citation_kind
from legal_rag.chunking.roles import ROLE_LABELS


class ContextBuilder:
    def __init__(self, max_chars: int = 1500, *, neighbor_radius: int = 0,
                 max_context_chars: int | None = None,
                 neighbor_corpora: list[str] | None = None,
                 neighbor_lookup: Callable[[Chunk, int], list[Chunk]] | None = None) -> None:
        if max_chars < 1 or (max_context_chars is not None and max_context_chars < 1) or not 0 <= neighbor_radius <= 2:
            raise ValueError("Context limits must be positive and neighbor radius between zero and two.")
        if neighbor_radius and neighbor_lookup is None:
            raise ValueError("Neighbor expansion requires a neighbor lookup.")
        self.max_chars = max_chars
        self.neighbor_radius = neighbor_radius
        self.max_context_chars = max_context_chars if max_context_chars is not None else (7500 if neighbor_radius else None)
        self.neighbor_lookup = neighbor_lookup
        self.neighbor_corpora = frozenset(neighbor_corpora or [])

    def build(self, hits: list[ScoredChunk]) -> EvidenceBundle:
        if not self.neighbor_radius and self.max_context_chars is None:
            return EvidenceBundle(items=[self._evidence(hit.chunk, f"S{n}", hit.score)
                                         for n, hit in enumerate(hits, 1)])
        items: list[Evidence] = []
        seen: set[tuple[str, str]] = set()
        anchors: list[tuple[Chunk, str]] = []

        def add(chunk: Chunk, score: float, anchor_id: str | None = None) -> bool:
            key = (chunk.metadata.corpus_version, chunk.id)
            if key in seen:
                return False
            item = self._evidence(chunk, f"S{len(items) + 1}", score)
            if anchor_id:
                item = item.model_copy(update={"header": {**item.header, "Context": f"Adjacent to {anchor_id}"}})
            # Bound actual rendered evidence including headers and source markers.
            if len(EvidenceBundle(items=[*items, item]).render()) > self.max_context_chars:
                return False
            items.append(item)
            seen.add(key)
            return True

        # All ranked anchors get first access to the budget; neighbors cannot crowd them out.
        for hit in hits:
            if add(hit.chunk, hit.score):
                anchors.append((hit.chunk, items[-1].source_id))
        for chunk, anchor_id in (anchors if self.neighbor_radius else []):
            if self.neighbor_corpora and chunk.metadata.corpus_version not in self.neighbor_corpora:
                continue
            for neighbor in self.neighbor_lookup(chunk, self.neighbor_radius):
                if (neighbor.document_id != chunk.document_id
                        or neighbor.metadata.corpus_version != chunk.metadata.corpus_version
                        or not 0 < abs(neighbor.order - chunk.order) <= self.neighbor_radius):
                    raise ValueError("Neighbor lookup returned a passage outside the requested document window.")
                add(neighbor, 0.0, anchor_id)
        return EvidenceBundle(items=items)

    def _evidence(self, chunk: Chunk, source_id: str, score: float) -> Evidence:
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
                "Section role": ROLE_LABELS[m.section_role],
            }
        else:
            header = {"Type": "Interpretation", "Authority": m.authority or "", "Reference number": m.reference_number or ""}
        header = {key: value for key, value in header.items() if value}
        text = chunk.text if len(chunk.text) <= self.max_chars else chunk.text[: self.max_chars] + "…"
        return Evidence(source_id=source_id, chunk_id=chunk.id, document_id=chunk.document_id,
                        document_type=chunk.document_type, header=header, text=text, score=score)

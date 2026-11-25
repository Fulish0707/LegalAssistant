"""Explicit offline fusion experiments; serving configuration is never modified."""
from __future__ import annotations
from dataclasses import dataclass, replace
from legal_rag.retrieval.fusion import RRFFuser
from legal_rag.schemas import ScoredChunk

@dataclass(frozen=True)
class FusionVariant:
    name: str
    leg_floor: int = 3
    budget: int = 100
    sparse_weight: float = .75
    rrf_k: int = 60
    alpha: float | None = None

    def fuse(self, lists: list[list[ScoredChunk]], *, budget: int | None = None) -> list[ScoredChunk]:
        k = self.budget if budget is None else budget
        weights = {h.retriever: self.sparse_weight if h.retriever.endswith(':sparse') else 1. for ranked in lists for h in ranked}
        fuser = RRFFuser(self.rrf_k, weights, self.leg_floor)
        if self.alpha is None:
            return fuser.fuse(lists, k)
        # Each input list belongs to one corpus and modality. Normalize before
        # combining, so incomparable raw scores never cross corpus boundaries.
        scores: dict[str, float] = {}
        chunks = {}
        active_lists = []
        for ranked in lists:
            if not ranked:
                continue
            low, high = min(h.score for h in ranked), max(h.score for h in ranked)
            active = []
            for h in ranked:
                weight = 1-self.alpha if h.retriever.endswith(':sparse') else self.alpha
                normalized = (h.score-low)/(high-low) if high != low else 1.
                scores[h.chunk.id] = scores.get(h.chunk.id, 0.) + weight * normalized
                chunks.setdefault(h.chunk.id, h.chunk)
                if weight > 0:
                    active.append(h)
            active_lists.append(active)
        chosen = fuser._floor(active_lists, k, {ident: 1. for ident in scores})
        for ident in sorted(scores, key=lambda i: (-scores[i], i)):
            if len(chosen) >= k:
                break
            chosen.setdefault(ident, None)
        return [ScoredChunk(chunk=chunks[i], score=scores[i], retriever='fused')
                for i in sorted(chosen, key=lambda i: (-scores[i], i))]

BASELINE = FusionVariant('baseline')
VARIANTS = {'baseline': BASELINE}
for name, changes in {
    'F0': {'leg_floor': 0}, 'F1': {'leg_floor': 5}, 'F2': {'budget': 200},
    'F3': {'sparse_weight': .5}, 'F4': {'sparse_weight': 1.},
    'F5': {'rrf_k': 20}, 'F6': {'rrf_k': 100},
    'F7a': {'alpha': .5}, 'F7b': {'alpha': .75}, 'F7c': {'alpha': 1.},
}.items():
    VARIANTS[name] = replace(BASELINE, name=name, **changes)

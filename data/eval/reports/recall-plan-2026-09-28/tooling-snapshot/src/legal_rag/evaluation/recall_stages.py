"""Request-local, serializable retrieval capture and offline replay.

No access to HybridRetriever.last_trace and no model-score cache. Exceptions
are explicit stage outcomes; consumers must exclude failed captures from recall.
"""
from __future__ import annotations
from dataclasses import asdict
from time import perf_counter
from typing import Any, Callable
from legal_rag.schemas import ScoredChunk
from .recall_variants import FusionVariant, VARIANTS


def _hits(hits: list[ScoredChunk]) -> list[dict[str, Any]]:
    return [dict(chunk_id=h.chunk.id, document_id=h.chunk.document_id,
                 corpus=h.retriever.rsplit(':', 1)[0] if ':' in h.retriever else h.chunk.metadata.dataset,
                 rank=rank, score=h.score, retriever=h.retriever,
                 chunk=h.chunk.model_dump(mode='json')) for rank, h in enumerate(hits, 1)]


def _stage(call: Callable[[], list[ScoredChunk]]) -> tuple[dict, list[ScoredChunk]]:
    start = perf_counter()
    try:
        hits = call()
        error = None
    except Exception as exc:
        hits = []
        error = {'type': type(exc).__name__, 'message': str(exc)}
    return {'hits': _hits(hits), 'timing_ms': (perf_counter()-start)*1000, 'error': error}, hits


def _preserve_corpora(legs: list[dict], *stages: dict) -> None:
    corpora = {hit['chunk_id']: leg['name'].rsplit(':', 1)[0]
               for leg in legs for hit in leg['hits']}
    for stage in stages:
        for hit in stage['hits']:
            hit['corpus'] = corpora[hit['chunk_id']]


def capture_stages(retriever, query: str, *, dense_k: int, sparse_k: int,
                   rerank_k: int = 100, filters=None) -> dict:
    """Call each real leg, fuser and reranker; return only request-owned state."""
    if min(dense_k, sparse_k, rerank_k) < 0:
        raise ValueError('Depths must be nonnegative')
    legs, lists = [], []
    for leg in retriever.legs:
        kind = leg.name.rsplit(':', 1)[-1]
        if kind not in ('dense', 'sparse'):
            raise ValueError(f'Unrecognized retrieval modality: {leg.name}')
        depth = dense_k if kind == 'dense' else sparse_k
        stage, hits = _stage(lambda: leg.retrieve(query, depth, filters))
        legs.append(dict(name=leg.name, kind=kind, requested_depth=depth, **stage))
        lists.append(hits)
    fused, candidates = _stage(lambda: retriever.fuser.fuse(lists, retriever.k_fused))
    reranked, _ = _stage(lambda: retriever.reranker.rerank(query, candidates, rerank_k))
    _preserve_corpora(legs, fused, reranked)
    return dict(schema_version=1, query=query, legs=legs, fused=fused, reranked=reranked,
                dense_k=dense_k, sparse_k=sparse_k, fused_k=retriever.k_fused,
                rerank_k=rerank_k, filters=filters.model_dump(mode='json') if filters is not None else None)


def capture_multi_stages(retriever, queries: list[str], *, depth: int,
                         rerank_k: int = 100, filters=None) -> dict:
    """Retrieve each query at the same per-leg depth; rerank with the original."""
    if not queries or any(not query.strip() for query in queries) or depth < 1 or rerank_k < 1:
        raise ValueError("nonempty queries and positive depths required")
    legs, ranked_lists = [], []
    for query_index, query in enumerate(queries):
        for leg in retriever.legs:
            kind = leg.name.rsplit(":", 1)[-1]
            if kind not in ("dense", "sparse"):
                raise ValueError(f"Unrecognized retrieval modality: {leg.name}")
            stage, hits = _stage(lambda leg=leg,query=query: leg.retrieve(query, depth, filters))
            legs.append(dict(name=leg.name, kind=kind, query_index=query_index,
                             requested_depth=depth, **stage))
            ranked_lists.append(hits)
    fused, candidates = _stage(lambda: retriever.fuser.fuse(ranked_lists, retriever.k_fused))
    reranked, _ = _stage(lambda: retriever.reranker.rerank(queries[0], candidates, rerank_k))
    _preserve_corpora(legs, fused, reranked)
    return dict(schema_version=1, query=queries[0], queries=queries, legs=legs,
                fused=fused, reranked=reranked, dense_k=depth, sparse_k=depth,
                fused_k=retriever.k_fused, rerank_k=rerank_k,
                filters=filters.model_dump(mode="json") if filters is not None else None)

def replay_stages(capture: dict, reranker, *, variant: FusionVariant | str = 'baseline',
                  rerank_k: int = 100, rerank: bool = True, full_union: bool = False) -> dict:
    """Replay stored chunk payloads using the original query and actual reranker.

    full_union increases fusion output to the distinct raw union size, preserving
    real fusion ordering before reranking. R0 is rerank=False (or variant='R0').
    """
    if variant == 'R0':
        variant, rerank = 'baseline', False
    selected = VARIANTS[variant] if isinstance(variant, str) else variant
    lists = [[ScoredChunk(chunk=h['chunk'], score=h['score'], retriever=h['retriever'])
              for h in leg['hits']] for leg in capture['legs']]
    budget = len({h.chunk.id for ranked in lists for h in ranked}) if full_union else selected.budget
    fused, candidates = _stage(lambda: selected.fuse(lists, budget=budget))
    reranked, _ = _stage(lambda: reranker.rerank(capture['query'], candidates, rerank_k)
                        if rerank else candidates[:rerank_k])
    _preserve_corpora(capture['legs'], fused, reranked)
    return dict(schema_version=1, query=capture['query'], variant=asdict(selected),
                full_union=full_union, rerank=rerank, fused=fused, reranked=reranked,
                input_errors=[dict(name=leg['name'], error=leg['error'])
                              for leg in capture['legs'] if leg['error'] is not None])

"""Case-level retrieval evidence; ranks refer to raw chunks, not deduplicated documents."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

from legal_rag.retrieval.hybrid import Retriever
from legal_rag.schemas import Chunk, RetrievalFilters


def normalize_question(text: str) -> str:
    """Normalize whitespace only; preserve wording, case, and punctuation."""
    return ' '.join(text.split())


def build_duplicate_question_index(rows: Iterable[dict]) -> dict[str, dict]:
    """Count rows and collect unique gold IDs for each normalized question."""
    index: dict[str, dict] = {}
    for row in rows:
        entry = index.setdefault(normalize_question(row['question']), {'count': 0, 'document_ids': set()})
        entry['count'] += 1
        entry['document_ids'].add(row['document_id'])
    return {question: {'count': entry['count'], 'document_ids': sorted(entry['document_ids'])} for question, entry in index.items()}


def _chunk_preview(chunk: Chunk, preview_chars: int) -> dict[str, Any]:
    date = chunk.metadata.decision_date or chunk.metadata.effective_date
    return {
        'chunk_id': chunk.id,
        'document_id': chunk.document_id,
        'document_type': str(chunk.document_type),
        'legal_domain': chunk.metadata.legal_domain,
        'chunk_index': chunk.order,
        'title': chunk.title,
        'authority': chunk.metadata.authority,
        'date': date.isoformat() if date else None,
        'text_preview': chunk.text[:preview_chars],
    }


def diagnose_case(
    qa_record: dict,
    retriever: Retriever,
    *,
    gold_chunks: list[Chunk],
    gold_chunk_count: int,
    duplicate_question_count: int,
    candidate_gold_document_ids: list[str],
    top_k: int,
    deep_k: int,
    preview_chars: int = 1200,
    max_gold_chunks: int = 10,
    case_id: str | None = None,
) -> dict[str, Any]:
    """Retrieve the question only and retain auditable chunk-level evidence.

    Default case IDs hash the canonical QA row except its answer, retaining
    review IDs when answer formatting changes. Exact/multi-positive hits are at top_k.
    Gold corpus presence is determined by the full count, not preview length.
    """
    if not 0 < top_k <= deep_k:
        raise ValueError('Require 0 < top_k <= deep_k')
    if preview_chars < 0 or max_gold_chunks < 0 or gold_chunk_count < 0:
        raise ValueError('Preview limits and gold chunk count must be nonnegative')
    if case_id is None:
        encoded = json.dumps({key: value for key, value in qa_record.items() if key != 'answer'}, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        case_id = hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:16]
    gold_id = qa_record['document_id']
    candidates = sorted(set(candidate_gold_document_ids) | {gold_id})
    present = gold_chunk_count > 0
    hits = []
    if present:
        domain = qa_record.get('legal_domain')
        filters = RetrievalFilters(legal_domain=domain) if domain else None
        hits = retriever.retrieve(qa_record['question'], deep_k, filters)[:deep_k]
    gold_rank = next((rank for rank, hit in enumerate(hits, 1) if hit.chunk.document_id == gold_id), None)
    top_hit = gold_rank is not None and gold_rank <= top_k
    deep_hit = gold_rank is not None
    trace = getattr(retriever, 'last_trace', {}) if present else {}
    trace = {name: [list(pair) for pair in pairs] for name, pairs in (trace or {}).items()}
    retrieved = []
    score_keys = {'dense': 'dense_score', 'sparse': 'sparse_score', 'fused': 'hybrid_score', 'reranked': 'rerank_score'}
    for rank, hit in enumerate(hits, 1):
        item = _chunk_preview(hit.chunk, preview_chars)
        item.update(rank=rank, retriever=hit.retriever)
        score_key = score_keys.get(hit.retriever.rsplit(':', 1)[-1])
        if score_key == 'rerank_score' and getattr(getattr(retriever, 'reranker', None), 'name', None) == 'noop':
            score_key = None
        if score_key:
            item[score_key] = hit.score
        for kind, key in (('dense', 'dense_score'), ('sparse', 'sparse_score'), ('fused', 'hybrid_score')):
            scores = [score for name, pairs in trace.items()
                      if name == kind or (name.endswith(':' + qa_record['question'])
                          and name[:-(len(qa_record['question']) + 1)].rsplit(':', 1)[-1] == kind)
                      for chunk_id, score in pairs if chunk_id == hit.chunk.id]
            # Multiple query/leg scores stay in the trace rather than being conflated.
            if len(scores) == 1 and key not in item:
                item[key] = scores[0]
        retrieved.append(item)
    return {
        'case_id': case_id,
        'status': 'ok' if present else 'missing_gold_corpus',
        'gold_document_id': gold_id,
        'gold_document_type': qa_record.get('document_type'),
        'question': qa_record['question'],
        'answer': qa_record.get('answer'),
        'legal_domain': qa_record.get('legal_domain'),
        'top_k': top_k,
        'deep_k': deep_k,
        'gold_rank': gold_rank,
        'rank_bucket': ('missing_gold_corpus' if not present else 'top_k' if top_hit else 'deep_candidate' if deep_hit else 'not_found'),
        'gold_found_in_top_k': top_hit,
        'gold_found_in_deep_k': deep_hit,
        'exact_gold_hit': top_hit,
        'multi_positive_hit': any(hit.chunk.document_id in candidates for hit in hits[:top_k]),
        'duplicate_question_count': duplicate_question_count,
        'candidate_gold_document_ids': candidates,
        'retrieved': retrieved,
        'retrieval_trace': trace,
        'gold_chunk_count': gold_chunk_count,
        'gold_chunks': [_chunk_preview(chunk, preview_chars) for chunk in gold_chunks[:max_gold_chunks]],
        'manual_category': None,
        'manual_relevant_result_ranks': [],
        'manual_notes': '',
    }

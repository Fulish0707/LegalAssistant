"""Extend an existing relevance pool with chunks that new runs surfaced, without retrieving again.

Re-fusing recorded leg runs under new weights (see :mod:`fusion_sweep`) promotes chunks the
source pool never held. Those pairs need pool rows before they can be judged, and the rows must
carry exactly the fields ``relevance_pool._pair`` writes, so the judge and the review CSV see no
difference between an original row and an extended one.
"""
from __future__ import annotations

from legal_rag.schemas import Chunk

from .relevance_pool import _pair


def gold_document_ids(pool_rows: list[dict], target_rows: list[dict] | None = None,
                      ) -> dict[str, set[str]]:
    """Gold document ids per case.

    A pool row flagged ``is_gold_chunk`` names its case's gold document, but a shallow pool holds
    such a row only for the few cases whose gold chunk was retrieved. ``target_rows`` (the
    diagnostic ``target.jsonl`` recorded in ``pool.json``) covers every case and wins; the pool's
    own gold rows are then a consistency check and must agree with it.
    """
    from_pool: dict[str, set[str]] = {}
    for row in pool_rows:
        if row.get('is_gold_chunk'):
            from_pool.setdefault(row['case_id'], set()).add(row['document_id'])
    if target_rows is None:
        return from_pool
    gold = {row['id']: set(row['relevant_document_ids']) for row in target_rows}
    for case_id, ids in from_pool.items():
        unknown = ids - gold.get(case_id, set())
        if unknown:
            raise ValueError(f'gold chunks pooled for {case_id} name documents {sorted(unknown)} that the '
                             'target file does not list; the target is stale or from another run')
    return gold


def extend_pool(pool_rows: list[dict], needed_pairs: dict[tuple[str, str], dict[str, int]],
                chunks: dict[str, Chunk], *, gold_documents: dict[str, set[str]]) -> list[dict]:
    """The pool for the new runs: source rows first, then a row per newly surfaced chunk.

    ``needed_pairs`` maps every pooled ``(case_id, chunk_id)`` to its ``{mode: rank}``, which
    replaces each row's ``ranks`` (a source row no run reaches keeps empty ranks, as an
    unretrieved gold chunk does). ``chunks`` must hold the corpus chunk for every pair the source
    pool does not already cover; a missing one raises rather than leaving an unjudgeable pair.
    """
    by_pair = {(row['case_id'], row['chunk_id']): row for row in pool_rows}
    cases: dict[str, dict] = {}
    for row in pool_rows:
        cases.setdefault(row['case_id'], row)
    extended = [dict(row, ranks=needed_pairs.get(pair, {})) for pair, row in by_pair.items()]
    for case_id, chunk_id in needed_pairs:
        if (case_id, chunk_id) in by_pair:
            continue
        case = cases.get(case_id)
        if case is None:
            raise ValueError(f'{case_id} is not a case of the source pool')
        chunk = chunks.get(chunk_id)
        if chunk is None:
            raise ValueError(f'no corpus chunk for {chunk_id} (case {case_id}); the pool row would be unjudgeable')
        row = _pair({'case_id': case_id, 'question': case['question'],
                     'answer': case['reference_answer'], 'document_id': None}, chunk)
        row['ranks'] = needed_pairs[(case_id, chunk_id)]
        row['is_gold_chunk'] = chunk.document_id in gold_documents.get(case_id, set())
        extended.append(row)
    return extended

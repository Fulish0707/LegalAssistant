"""Aggregate diagnostic observations and write immutable raw / editable review files."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

MANUAL_CATEGORIES = {
    'A': 'True Retrieval Failure: relevant evidence exists but is absent from candidates.',
    'B': 'Evaluation Label Ambiguity: an alternative retrieved source is sufficient.',
    'C': 'Source / Preprocessing Problem: expected evidence is absent from processed chunks.',
    'D': 'Ranking Problem: relevant evidence is in deep candidates but outside Top-K.',
    'E': 'Chunking Problem: evidence is poorly split or mixed with unrelated text.',
}


def summarize_cases(cases: Iterable[dict], *, legal_domain: str, retrieval_mode: str,
                    top_k: int, deep_k: int) -> dict:
    """Metrics use all evaluated rows, before failure filtering or review sampling.

    Ranks count returned chunks, including repeated documents. Missing corpus golds
    have their own denominator and never depress retrieval metrics.
    """
    cases = list(cases)
    evaluated = [c for c in cases if c['status'] != 'missing_gold_corpus']
    n = len(evaluated)
    ranks = [c['gold_rank'] for c in evaluated]
    counts = {
        'gold_rank_le_top_k': sum(r is not None and r <= top_k for r in ranks),
        'gold_rank_between_top_k_and_deep_k': sum(r is not None and top_k < r <= deep_k for r in ranks),
        'gold_not_found_within_deep_k': sum(r is None for r in ranks),
    }
    return {
        'legal_domain': legal_domain, 'retrieval_mode': retrieval_mode,
        'evaluated': n, 'target_cases': len(cases),
        'missing_gold_corpus': len(cases) - n,
        'top_k': top_k, 'deep_k': deep_k, 'rank_unit': 'retrieved_chunk',
        'exact_recall_at_k': counts['gold_rank_le_top_k'] / n if n else None,
        'multi_positive_recall_at_k': sum(c['multi_positive_hit'] for c in evaluated) / n if n else None,
        **counts,
        **{f'{key}_percentage': value * 100 / n if n else None for key, value in counts.items()},
        'duplicate_question_cases': sum(c['duplicate_question_count'] > 1 for c in evaluated),
        'recall_at': {str(k): sum(r is not None and r <= k for r in ranks) / n if n else None
                      for k in sorted({5, 10, 20, 50, 100, top_k, deep_k}) if k <= deep_k},
        'mrr': sum(1/r for r in ranks if r is not None) / n if n else None,
        'mrr_depth': deep_k,
    }


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open('x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_diagnostic_reports(output: Path, cases: list[dict], missing: list[dict], summary: dict) -> Path:
    """Refuse an existing run directory: raw evidence and annotations are never replaced."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output / 'failures.jsonl', cases)
    _write_jsonl(output / 'missing_gold_corpus.jsonl', missing)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    fields = ['case_id', 'question', 'answer', 'gold_document_id', 'gold_document_type',
              'gold_rank', 'gold_found_in_top_k', 'gold_found_in_deep_k',
              'duplicate_question_count', 'candidate_gold_document_ids', 'multi_positive_hit',
              'gold_chunk_count', 'gold_chunk_previews']
    for rank in range(1, 4):
        fields.extend(f'top_{rank}_{key}' for key in ('document_id', 'document_type', 'text_preview'))
    fields.extend(('manual_category', 'manual_relevant_result_ranks', 'manual_notes'))
    with (output / 'review.csv').open('x', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            row = {key: case.get(key) for key in fields}
            row['candidate_gold_document_ids'] = json.dumps(case.get('candidate_gold_document_ids', []), ensure_ascii=False)
            row['manual_relevant_result_ranks'] = json.dumps(case.get('manual_relevant_result_ranks', []))
            row['gold_chunk_previews'] = '\n\n'.join(chunk['text_preview'] for chunk in case.get('gold_chunks', [])[:3])
            for rank, hit in enumerate(case['retrieved'][:3], 1):
                for key in ('document_id', 'document_type', 'text_preview'):
                    row[f'top_{rank}_{key}'] = hit.get(key)
            writer.writerow(row)
    categories = '\n'.join(f'- {key}: {value}' for key, value in MANUAL_CATEGORIES.items())
    (output / 'README.md').write_text(
        '# Retrieval diagnostic review\n\n'
        'Edit manual_category (A–E), manual_relevant_result_ranks (JSON array), and manual_notes in review.csv. '
        'Raw JSONL remains unchanged. Use case_id to join annotations to evidence. '
        'No category or legal sufficiency judgment is automated.\n\n'
        + categories + '\n\n'
        'failures.jsonl contains selected review cases (including successes when only_failures=false). '
        'missing_gold_corpus.jsonl is separate and excluded from recall. '
        'summary.json metrics cover the full evaluated target, before review sampling. '
        'Rank counts chunks, not distinct documents; MRR is truncated at deep_k.\n', encoding='utf-8')
    return output

"""Draw a reproducible evaluation target from domains other than the one already measured.

Case ids come from :func:`legal_rag.evaluation.diagnostic_runner.load_diagnostic_rows`, so a
written ``target.jsonl`` reproduces exactly the drawn cases when it is loaded back with that
domain's ``qa.jsonl``. That only holds when each drawn row owns its (normalized question,
document_id) pair outright, because the dataset match is by pair and not by case id, hence
:func:`drop_ambiguous_pairs`.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
from collections import Counter
from typing import Callable

from .diagnostics import normalize_question


def drop_ambiguous_pairs(rows: list[dict]) -> list[dict]:
    """Rows whose (normalized question, document_id) pair no other row in ``rows`` shares."""
    pairs = Counter((normalize_question(row['question']), row['document_id']) for row in rows)
    return [row for row in rows if pairs[(normalize_question(row['question']), row['document_id'])] == 1]


def sample_cases(rows: list[dict], *, limit: int, seed: int,
                 has_chunks: Callable[[str], bool]) -> list[dict]:
    """Draw ``limit`` rows whose gold document has chunks, shuffling under ``seed``.

    Eligibility is checked in the shuffled order and only until the quota is filled, so the draw
    does not depend on how many ineligible rows follow it. The result is sorted by case id.
    """
    if limit < 1:
        raise ValueError('limit must be positive')
    shuffled = sorted(rows, key=lambda row: row['case_id'])
    random.Random(seed).shuffle(shuffled)
    drawn: list[dict] = []
    for row in shuffled:
        if has_chunks(row['document_id']):
            drawn.append(row)
            if len(drawn) == limit:
                return sorted(drawn, key=lambda item: item['case_id'])
    raise ValueError(f'only {len(drawn)} of {len(rows)} rows have gold chunks in the corpus; {limit} were asked for')


def target_rows(cases: list[dict]) -> list[dict]:
    """Rows in the shape ``diagnostic_runner`` writes, so the same loader reproduces the cases."""
    return [{'id': case['case_id'], 'query': case['question'], 'relevant_document_ids': [case['document_id']]}
            for case in cases]


def sidecar_rows(cases: list[dict], corpora: dict[str, str]) -> list[dict]:
    """Each case's domain and the corpus that holds its gold document; target.jsonl records neither."""
    return [{'id': case['case_id'], 'legal_domain': case['legal_domain'],
             'corpus_version': corpora[case['legal_domain']], 'document_id': case['document_id']}
            for case in cases]


def corpus_document_ids(directory: Path) -> set[str]:
    """Every document id with at least one chunk in a processed corpus.

    Chunks are read as plain JSON rather than validated: this streams gigabytes only to answer
    which gold documents were indexed, and the selected golds are validated later anyway.
    """
    directory = Path(directory)
    found: set[str] = set()
    with (directory / 'chunks.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                found.add(json.loads(line)['document_id'])
    if not found:
        raise ValueError(f'no processed chunks at {directory}')
    return found

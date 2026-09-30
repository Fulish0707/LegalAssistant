"""Pairwise answer comparison between two corpora over the same questions.

Retrieval metrics said ip-v1 and ip-v2 rank equally well, but they differ in what reaches the
generator: the same ten slots carry 24k characters of ip-v1 chunks against 7.6k of ip-v2. That
gap only shows in the answer, so the answers are compared directly: one question, two answers,
a judge that picks the better one without knowing which corpus produced it.

Sides are randomised per case, so a judge that always prefers the first answer scores 50%.
"""
from __future__ import annotations

import random
import statistics
from typing import Iterable

from legal_rag.schemas import Evidence, EvidenceBundle

CSV_FIELDS = ['case_id', 'question', 'answer_a', 'answer_b', 'winner', 'reason']
# The judge reads the reference before the two answers it is asked to compare.
CSV_FIELDS_WITH_REFERENCE = ['case_id', 'question', 'reference_answer', 'answer_a', 'answer_b', 'winner', 'reason']
WINNERS = ('A', 'B', 'tie')


def evidence_bundle(pool: dict[tuple[str, str], dict], case_id: str, ranked: list[dict], k: int) -> EvidenceBundle:
    """The top ``k`` retrieved chunks as the generator sees them."""
    items = []
    for rank, hit in enumerate(ranked[:k], 1):
        row = pool.get((case_id, hit['chunk_id']))
        if row is None:
            continue
        header = {'문서': row.get('case_number') or row['document_id'], '유형': row.get('document_type', '')}
        # The recorded ranking carries no score; rank order is what the generator sees anyway.
        items.append(Evidence(source_id=f'S{rank}', text=row['text'], header=header, score=0.0,
                              chunk_id=row['chunk_id'], document_id=row['document_id'],
                              document_type=row['document_type']))
    return EvidenceBundle(items=items)


def pairwise_rows(answers: list[dict], *, seed: int) -> tuple[list[dict], dict[str, str]]:
    """CSV rows with the two answers randomly assigned to A and B, plus the hidden mapping.

    ``answers`` rows carry ``case_id``, ``question``, and one answer per side name, and optionally
    a ``reference_answer`` that is copied into the row for the judge. The mapping is written
    separately so the judge never sees which side wrote which answer.
    """
    rows, key = [], {}
    for answer in sorted(answers, key=lambda row: row['case_id']):
        sides = sorted(name for name in answer if name not in ('case_id', 'question', 'reference_answer'))
        if len(sides) != 2:
            raise ValueError(f"{answer['case_id']}: expected two answers, got {sides}")
        # Seeded per case, not per sequence: a case keeps its sides when the sample around it
        # changes, so verdicts collected on an earlier, smaller run stay valid.
        rng = random.Random(f"{seed}:{answer['case_id']}")
        first, second = (sides if rng.random() < 0.5 else sides[::-1])
        key[answer['case_id']] = f'A={first},B={second}'
        row = {'case_id': answer['case_id'], 'question': answer['question'],
               'answer_a': answer[first], 'answer_b': answer[second], 'winner': '', 'reason': ''}
        if 'reference_answer' in answer:
            row['reference_answer'] = answer['reference_answer']
        rows.append(row)
    return rows, key


def win_rate(verdicts: dict[str, str], key: dict[str, str], *, side: str, seed: int = 0,
             samples: int = 4000) -> dict:
    """How often ``side`` won, ties counted as half, with a bootstrap interval over cases."""
    scores = []
    for case_id, winner in sorted(verdicts.items()):
        if winner not in WINNERS:
            raise ValueError(f'{case_id}: winner must be one of {WINNERS}, got {winner!r}')
        mapping = dict(part.split('=', 1) for part in key[case_id].split(','))
        scores.append(0.5 if winner == 'tie' else float(mapping[winner] == side))
    if not scores:
        raise ValueError('no verdicts')
    rng = random.Random(seed)
    boots = sorted(statistics.mean(rng.choices(scores, k=len(scores))) for _ in range(samples))
    lower, upper = boots[int(samples * 0.025)], boots[int(samples * 0.975) - 1]
    return {'cases': len(scores), 'side': side, 'win_rate': statistics.mean(scores),
            'wins': sum(score == 1 for score in scores), 'losses': sum(score == 0 for score in scores),
            'ties': sum(score == 0.5 for score in scores), 'ci': [lower, upper],
            'decisive': lower > 0.5 or upper < 0.5}


def render_answer(answer) -> str:
    """The parts of a ``LegalAnswer`` a reader judges, as plain text."""
    lines = [answer.summary, '', answer.analysis]
    if answer.legal_issues:
        lines += ['', '쟁점: ' + '; '.join(answer.legal_issues)]
    if answer.possible_actions:
        lines += ['대응: ' + '; '.join(answer.possible_actions)]
    if answer.citations:
        lines += ['근거: ' + '; '.join(f'{c.source_id} {c.quote[:60]}' if getattr(c, 'quote', '') else c.source_id
                                      for c in answer.citations)]
    if answer.uncertainty:
        lines += ['불확실: ' + answer.uncertainty]
    return '\n'.join(lines).strip()


def sample_cases(case_ids: Iterable[str], *, limit: int, seed: int) -> list[str]:
    ordered = sorted(case_ids)
    if limit >= len(ordered):
        return ordered
    return sorted(random.Random(seed).sample(ordered, limit))

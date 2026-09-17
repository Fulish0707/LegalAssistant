"""Stratified human calibration of the relevance judge.

Pairs are drawn before any LLM grade is visible, stratified so the sample is not nearly all
grade 0. The CSV leads with a searchable case number and hides strata, ranks and gold status.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

STRATA = {'gold': 10, 'rank_1_3': 15, 'rank_4_10': 15, 'rank_11_20': 10}
CSV_FIELDS = ['pair_id', 'case_number', 'search_query', 'authority', 'date', 'title', 'document_type',
              'document_id', 'question', 'reference_answer', 'chunk_text', 'human_grade', 'human_note']


def _stratum(row: dict) -> str | None:
    if row['is_gold_chunk']:
        return 'gold'
    best = min(row['ranks'].values(), default=None)
    if best is None:
        return None
    return 'rank_1_3' if best <= 3 else 'rank_4_10' if best <= 10 else 'rank_11_20' if best <= 20 else None


def sample_calibration(pool: list[dict], *, seed: int, quotas: dict[str, int] = STRATA) -> list[dict]:
    rng = random.Random(seed)
    ordered = sorted(pool, key=lambda row: (row['case_id'], row['chunk_id']))
    sample = []
    for name, quota in quotas.items():
        members = [row for row in ordered if _stratum(row) == name]
        if len(members) < quota:
            raise ValueError(f'stratum {name} has {len(members)} pairs, needs {quota}')
        sample += [dict(row, stratum=name) for row in rng.sample(members, quota)]
    return sample


def _pair_id(row: dict) -> str:
    return f"{row['case_id']}|{row['chunk_id']}"


def write_calibration(output: Path, sample: list[dict]) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, csv_path = output / 'calibration.json', output / 'calibration.csv'
    # Refuse as a unit so an existing file never leaves a half-written calibration behind.
    for path in (manifest_path, csv_path):
        if path.exists():
            raise FileExistsError(path)
    # Ordering by pair id keeps each question's pairs together for the labeller; it is
    # independent of stratum, rank and gold status, so row order does not reveal them.
    rows = sorted(sample, key=_pair_id)
    with manifest_path.open('x', encoding='utf-8') as handle:
        json.dump({'pairs': [{'pair_id': _pair_id(r), 'stratum': r['stratum']} for r in rows]}, handle,
                  ensure_ascii=False, indent=2)
    with csv_path.open('x', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{key: row.get(key) for key in CSV_FIELDS}, 'pair_id': _pair_id(row),
                             'chunk_text': row['text'], 'human_grade': '', 'human_note': ''})
    return output


def read_human_grades(path: Path) -> dict[tuple[str, str], int]:
    grades = {}
    try:
        with Path(path).open(newline='', encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
    except UnicodeDecodeError as error:
        raise ValueError(f'{path} is not UTF-8 ({error}); re-save it from Excel as "CSV UTF-8"') from error
    for row in rows:
        value = (row.get('human_grade') or '').strip()
        if not value:
            continue
        if value not in {'0', '1', '2'}:
            raise ValueError(f"{row['pair_id']}: human_grade must be 0, 1 or 2, got {value!r}")
        case_id, chunk_id = row['pair_id'].split('|', 1)
        grades[(case_id, chunk_id)] = int(value)
    return grades


def _kappa(a: list[int], b: list[int], weighted: bool) -> float | None:
    labels = sorted(set(a) | set(b) | ({0, 1, 2} if weighted else set()))
    n = len(a)
    if not n:
        return None
    weight = (lambda i, j: abs(i - j) / (len(labels) - 1)) if weighted else (lambda i, j: float(i != j))
    observed = sum(weight(x, y) for x, y in zip(a, b)) / n
    expected = sum(a.count(i) * b.count(j) * weight(i, j) for i in labels for j in labels) / n ** 2
    return None if expected == 0 else 1 - observed / expected


def agreement(human: dict, llm: dict, *, gold_pairs: set) -> dict:
    pairs = sorted(key for key in human if key in llm)
    h, l = [human[k] for k in pairs], [llm[k] for k in pairs]
    confusion = [[sum(x == i and y == j for x, y in zip(h, l)) for j in range(3)] for i in range(3)]
    bh, bl = [int(x == 2) for x in h], [int(y == 2) for y in l]
    tp = sum(x and y for x, y in zip(bh, bl))
    return {
        'pairs': len(pairs), 'missing_llm': len([k for k in human if k not in llm]),
        'confusion': confusion,
        'exact_agreement': sum(x == y for x, y in zip(h, l)) / len(pairs) if pairs else None,
        'weighted_kappa': _kappa(h, l, weighted=True),
        'binary_grade2': {'kappa': _kappa(bh, bl, weighted=False),
                          'precision': tp / sum(bl) if sum(bl) else None,
                          'recall': tp / sum(bh) if sum(bh) else None},
        'gold_llm_grades': {str(g): sum(llm[k] == g for k in pairs if k in gold_pairs) for g in range(3)},
    }


PASS_KAPPA = 0.6
REVIEW_FIELDS = ['pair_id', 'case_number', 'search_query', 'human_grade', 'llm_grade', 'llm_reason', 'human_note']


def require_human_grades(pool_dir: Path) -> int:
    """Raise ``ValueError`` unless every pair in calibration.json has a human grade in calibration.csv.

    The judge must not run before the human has labelled the calibration set blind."""
    pool_dir = Path(pool_dir)
    manifest_path, csv_path = pool_dir / 'calibration.json', pool_dir / 'calibration.csv'
    for path in (manifest_path, csv_path):
        if not path.exists():
            raise ValueError(f'{path} is missing; run sample_calibration.py and grade it first')
    wanted = {pair['pair_id'] for pair in json.loads(manifest_path.read_text(encoding='utf-8'))['pairs']}
    graded = {f'{case_id}|{chunk_id}' for case_id, chunk_id in read_human_grades(csv_path)}
    missing = wanted - graded
    if missing:
        raise ValueError(f'{len(missing)} of {len(wanted)} calibration pairs have no human_grade in {csv_path}; '
                         'finish labelling before judging')
    return len(wanted)


def judge_passes(result: dict, threshold: float = PASS_KAPPA) -> bool:
    """Pass only when every human-graded pair has a judge grade and binary grade-2 kappa meets the threshold."""
    kappa = result['binary_grade2']['kappa']
    return result['missing_llm'] == 0 and kappa is not None and kappa >= threshold


def missing_llm_message(result: dict) -> str:
    return (f"{result['missing_llm']} calibration pairs have no judge grade (failed parse or not judged); "
            'run judge_relevance.py --calibration-only [--retry-failed]')


def disagreement_rows(csv_path: Path, llm_rows: dict[tuple[str, str], dict]) -> list[dict]:
    """Human-graded pairs whose stored judgment differs from the human grade (a failed parse,
    grade None, counts as a disagreement). Pairs the judge has not reached are left out."""
    with Path(csv_path).open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    human = read_human_grades(csv_path)
    review = []
    for row in rows:
        key = tuple(row['pair_id'].split('|', 1))
        if key not in human or key not in llm_rows or llm_rows[key]['grade'] == human[key]:
            continue
        judged = llm_rows[key]
        review.append({'pair_id': row['pair_id'], 'case_number': row.get('case_number'),
                       'search_query': row.get('search_query'), 'human_grade': human[key],
                       'llm_grade': judged['grade'], 'llm_reason': judged.get('reason'),
                       'human_note': row.get('human_note')})
    return review


def write_review(path: Path, rows: list[dict]) -> Path:
    with Path(path).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return Path(path)

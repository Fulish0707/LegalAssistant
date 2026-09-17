"""Stratified calibration of the relevance judge against reference labels (codex_grade).

Pairs are drawn before any LLM grade is visible, stratified so the sample is not nearly all
grade 0. The CSV leads with a searchable case number and hides strata, ranks and gold status.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

STRATA = {'gold': 10, 'rank_1_3': 15, 'rank_4_10': 15, 'rank_11_20': 10}
CSV_FIELDS = ['pair_id', 'case_number', 'search_query', 'authority', 'date', 'title', 'document_type',
              'document_id', 'question', 'reference_answer', 'chunk_text', 'codex_grade', 'codex_note']
# Reference-label columns, newest name first; calibration files written before the rename use human_*.
GRADE_COLUMNS = ('codex_grade', 'human_grade')
NOTE_COLUMNS = ('codex_note', 'human_note')


def _stratum(row: dict, *, gold_stratum: bool = True) -> str | None:
    """Gold chunks form their own stratum when sampled as one; otherwise they are placed by rank
    like any other pair (so a retrieved gold chunk can still be drawn)."""
    if gold_stratum and row['is_gold_chunk']:
        return 'gold'
    best = min(row['ranks'].values(), default=None)
    if best is None:
        return None
    return 'rank_1_3' if best <= 3 else 'rank_4_10' if best <= 10 else 'rank_11_20' if best <= 20 else None


def sample_calibration(pool: list[dict], *, seed: int, quotas: dict[str, int] = STRATA,
                       exclude: set[str] | frozenset = frozenset()) -> list[dict]:
    """Draw ``quotas`` pairs per stratum, in the quotas' order, skipping pair ids in ``exclude``."""
    unknown = set(quotas) - set(STRATA)
    if unknown:
        raise ValueError(f'unknown stratum {sorted(unknown)}; choose from {list(STRATA)}')
    rng = random.Random(seed)
    ordered = sorted((row for row in pool if _pair_id(row) not in exclude),
                     key=lambda row: (row['case_id'], row['chunk_id']))
    gold_stratum = 'gold' in quotas
    sample = []
    for name, quota in quotas.items():
        members = [row for row in ordered if _stratum(row, gold_stratum=gold_stratum) == name]
        if len(members) < quota:
            raise ValueError(f'stratum {name} has {len(members)} pairs, needs {quota}')
        sample += [dict(row, stratum=name) for row in rng.sample(members, quota)]
    return sample


def parse_quotas(values: list[str]) -> dict[str, int]:
    """``['rank_1_3=25', ...]`` to an ordered quota dict; no values means the default STRATA."""
    if not values:
        return dict(STRATA)
    quotas: dict[str, int] = {}
    for value in values:
        name, sep, count = value.partition('=')
        if not sep or not count.strip().isdigit() or int(count) < 1:
            raise ValueError(f'quota must look like rank_1_3=25, got {value!r}')
        if name not in STRATA:
            raise ValueError(f'unknown stratum {name!r}; choose from {list(STRATA)}')
        if name in quotas:
            raise ValueError(f'stratum {name!r} given twice')
        quotas[name] = int(count)
    return quotas


def calibration_pair_ids(pool_dir: Path) -> set[str]:
    """Pair ids in a pool directory's calibration.json (read only)."""
    manifest = json.loads((Path(pool_dir) / 'calibration.json').read_text(encoding='utf-8'))
    return {pair['pair_id'] for pair in manifest['pairs']}


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
                             'chunk_text': row['text'], 'codex_grade': '', 'codex_note': ''})
    return output


def _cell(row: dict, columns: tuple[str, ...]) -> str:
    """The one filled value among ``columns``; two different filled values are a labelling error."""
    values = {(row.get(column) or '').strip() for column in columns} - {''}
    if len(values) > 1:
        raise ValueError(f"{row.get('pair_id')}: conflicting values in {', '.join(columns)}: {sorted(values)}")
    return values.pop() if values else ''


def read_reference_grades(path: Path) -> dict[tuple[str, str], int]:
    grades = {}
    try:
        with Path(path).open(newline='', encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
    except UnicodeDecodeError as error:
        raise ValueError(f'{path} is not UTF-8 ({error}); re-save it from Excel as "CSV UTF-8"') from error
    for row in rows:
        value = _cell(row, GRADE_COLUMNS)
        if not value:
            continue
        if value not in {'0', '1', '2'}:
            raise ValueError(f"{row['pair_id']}: codex_grade must be 0, 1 or 2, got {value!r}")
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


def _binary(h: list[int], l: list[int], *, boundary: int) -> dict:
    bh, bl = [int(x >= boundary) for x in h], [int(y >= boundary) for y in l]
    tp = sum(x and y for x, y in zip(bh, bl))
    return {'kappa': _kappa(bh, bl, weighted=False),
            'precision': tp / sum(bl) if sum(bl) else None,
            'recall': tp / sum(bh) if sum(bh) else None}


def agreement(reference: dict, llm: dict, *, gold_pairs: set) -> dict:
    pairs = sorted(key for key in reference if key in llm)
    h, l = [reference[k] for k in pairs], [llm[k] for k in pairs]
    confusion = [[sum(x == i and y == j for x, y in zip(h, l)) for j in range(3)] for i in range(3)]
    return {
        'pairs': len(pairs), 'missing_llm': len([k for k in reference if k not in llm]),
        'confusion': confusion,
        'exact_agreement': sum(x == y for x, y in zip(h, l)) / len(pairs) if pairs else None,
        'weighted_kappa': _kappa(h, l, weighted=True),
        'binary_relevant': _binary(h, l, boundary=1),
        'binary_grade2': _binary(h, l, boundary=2),
        'gold_llm_grades': {str(g): sum(llm[k] == g for k in pairs if k in gold_pairs) for g in range(3)},
    }


PASS_KAPPA = 0.6
REVIEW_FIELDS = ['pair_id', 'case_number', 'search_query', 'codex_grade', 'llm_grade', 'llm_reason',
                 'evidence_quote', 'quote_verified', 'model_grade', 'codex_note']


def agreement_paths(pool_dir: Path, prompt_sha256: str) -> tuple[Path, Path]:
    """Per-key agreement report and review CSV, so runs with different prompts or think modes
    never overwrite each other (or the unsuffixed files of the first calibration)."""
    prefix = prompt_sha256[:12]
    return Path(pool_dir) / f'agreement-{prefix}.json', Path(pool_dir) / f'calibration_review-{prefix}.csv'


def model_grades(llm_rows: dict[tuple[str, str], dict]) -> dict[tuple[str, str], int]:
    """The judge's own grade before quote verification; rows without details (prompt v1) use the stored grade.
    Parse failures are left out."""
    grades = {}
    for key, row in llm_rows.items():
        grade = (row.get('details') or {}).get('model_grade', row['grade'])
        if grade is not None:
            grades[key] = grade
    return grades


def quote_downgrades(llm_rows: dict[tuple[str, str], dict], pairs) -> int:
    """Pairs among ``pairs`` whose model grade 2 was stored as 1 because the evidence quote failed verification."""
    return sum(1 for key in pairs if key in llm_rows and llm_rows[key]['grade'] == 1
               and (llm_rows[key].get('details') or {}).get('model_grade') == 2)


def require_reference_grades(pool_dir: Path) -> int:
    """Raise ``ValueError`` unless every pair in calibration.json has a reference grade in calibration.csv.

    The judge must not run before the reference labeller has graded the calibration set blind."""
    pool_dir = Path(pool_dir)
    manifest_path, csv_path = pool_dir / 'calibration.json', pool_dir / 'calibration.csv'
    for path in (manifest_path, csv_path):
        if not path.exists():
            raise ValueError(f'{path} is missing; run sample_calibration.py and grade it first')
    wanted = {pair['pair_id'] for pair in json.loads(manifest_path.read_text(encoding='utf-8'))['pairs']}
    graded = {f'{case_id}|{chunk_id}' for case_id, chunk_id in read_reference_grades(csv_path)}
    missing = wanted - graded
    if missing:
        raise ValueError(f'{len(missing)} of {len(wanted)} calibration pairs have no codex_grade in {csv_path}; '
                         'finish labelling before judging')
    return len(wanted)


def calibration_fingerprint(pool_dir: Path) -> dict[str, str | None]:
    """SHA-256 of the reference labels, the calibration manifest and the pool (None when a file is missing),
    so an agreement result or judge run can be tied to the exact labels it was checked against."""
    pool_dir = Path(pool_dir)

    def sha(name: str) -> str | None:
        path = pool_dir / name
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

    return {'calibration_csv_sha256': sha('calibration.csv'), 'calibration_json_sha256': sha('calibration.json'),
            'pool_jsonl_sha256': sha('pool.jsonl')}


BOUNDARY_KEYS = {1: 'binary_relevant', 2: 'binary_grade2'}


def boundary_kappa(result: dict, boundary: int) -> float | None:
    """Binary kappa at grade >= ``boundary`` (1: relevant vs 0, 2: grade 2 vs 0/1)."""
    if boundary not in BOUNDARY_KEYS:
        raise ValueError(f'boundary must be 1 or 2, got {boundary!r}')
    return result[BOUNDARY_KEYS[boundary]]['kappa']


def judge_passes(result: dict, boundary: int = 2, *, threshold: float = PASS_KAPPA) -> bool:
    """Pass only when every reference-graded pair has a judge grade and binary kappa at grade >= ``boundary``
    meets the threshold."""
    kappa = boundary_kappa(result, boundary)
    return result['missing_llm'] == 0 and kappa is not None and kappa >= threshold


def missing_llm_message(result: dict) -> str:
    return (f"{result['missing_llm']} calibration pairs have no judge grade (failed parse or not judged); "
            'run judge_relevance.py --calibration-only [--retry-failed]')


def disagreement_rows(csv_path: Path, llm_rows: dict[tuple[str, str], dict]) -> list[dict]:
    """Reference-graded pairs whose stored judgment differs from the reference grade (a failed parse,
    grade None, counts as a disagreement). Pairs the judge has not reached are left out."""
    with Path(csv_path).open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    reference = read_reference_grades(csv_path)
    review = []
    for row in rows:
        key = tuple(row['pair_id'].split('|', 1))
        if key not in reference or key not in llm_rows or llm_rows[key]['grade'] == reference[key]:
            continue
        judged = llm_rows[key]
        details = judged.get('details') or {}
        review.append({'pair_id': row['pair_id'], 'case_number': row.get('case_number'),
                       'search_query': row.get('search_query'), 'codex_grade': reference[key],
                       'llm_grade': judged['grade'], 'llm_reason': judged.get('reason'),
                       'evidence_quote': details.get('evidence_quote'), 'quote_verified': details.get('quote_verified'),
                       'model_grade': details.get('model_grade'), 'codex_note': _cell(row, NOTE_COLUMNS)})
    return review


def write_review(path: Path, rows: list[dict]) -> Path:
    with Path(path).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return Path(path)

"""Markdown and CSV rendering of graded retrieval metrics."""
from __future__ import annotations

import csv
from pathlib import Path

from .graded_metrics import LEVELS

SUMMARY_ROWS = ['cases', 'cases_without_relevant', 'ndcg@10', 'success@5', 'success@10', 'success@20', 'mrr',
                'precision@10_grade1', 'precision@10_grade2', 'mean_judged_coverage@10', 'min_judged_coverage@10']
CASE_METRICS = ['ndcg@10', 'success@10', 'mrr', 'judged_coverage@10']


def min_chunk_coverage(summary: dict) -> float:
    return min(levels['chunk']['min_judged_coverage@10'] for levels in summary.values())


COUNT_ROWS = {'cases', 'cases_without_relevant'}


def _cell(row: str, value) -> str:
    if value is None:
        return '—'
    return str(value) if row in COUNT_ROWS else f'{value:.3f}'


def render_markdown(summary: dict, *, judge: dict, pool: dict) -> str:
    """``summary`` is ``{mode: {level: metrics}}``; ``judge`` has model, prompt_sha256, failed_parses;
    ``pool`` is pool.json."""
    modes = list(summary)
    lines = [
        '# Graded retrieval comparison', '',
        f"- Judge: `{judge['model']}`, prompt `{judge['prompt_sha256'][:12]}`, failed parses: {judge['failed_parses']}",
        f"- Pool: corpus_version `{pool.get('corpus_version')}`, filters `{pool.get('filters')}`, "
        f"pool_depth {pool.get('pool_depth')}, settings_hash `{pool.get('settings_hash')}`",
        f"- Scoped to {pool.get('corpus_version')} with filter {pool.get('filters')}; "
        'multi-corpus fan-out is not measured.',
    ]
    for level in LEVELS:
        lines += ['', f'## {level}', '', '| Metric | ' + ' | '.join(modes) + ' |', '|---|' + '---:|' * len(modes)]
        lines += [f'| {row} | ' + ' | '.join(_cell(row, summary[m][level][row]) for m in modes) + ' |'
                  for row in SUMMARY_ROWS]
    if min_chunk_coverage(summary) < 1:
        lines += ['', '**WARNING:** some top-10 chunks are unjudged; run judge_relevance.py first. '
                  'Metrics treat them as grade 0 and the comparison is not valid.']
    return '\n'.join(lines) + '\n'


def case_rows(per_case: dict, questions: dict[str, str]) -> tuple[list[str], list[dict]]:
    """Flatten ``per_case_metrics`` output to one row per case with ``{mode}.{level}.{metric}`` columns."""
    fields = ['case_id', 'question'] + [f'{mode}.{level}.{metric}' for mode in per_case for level in LEVELS
                                        for metric in CASE_METRICS]
    rows: dict[str, dict] = {}
    for mode, levels in per_case.items():
        for level in LEVELS:
            for result in levels[level]:
                row = rows.setdefault(result['case_id'], {'case_id': result['case_id'],
                                                          'question': questions.get(result['case_id'], '')})
                row.update({f'{mode}.{level}.{metric}': result[metric] for metric in CASE_METRICS})
    return fields, sorted(rows.values(), key=lambda row: row['case_id'])


def write_cases_csv(path: Path, fields: list[str], rows: list[dict]) -> Path:
    with Path(path).open('x', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return Path(path)

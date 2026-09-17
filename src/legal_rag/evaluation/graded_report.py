"""Markdown and CSV rendering of graded retrieval metrics."""
from __future__ import annotations

import csv
from pathlib import Path

from .graded_metrics import LEVELS


def _is_case_metric(name: str) -> bool:
    return (name in ('ndcg@10', 'ndcg@10_binary', 'success@10', 'precision@10', 'failed@10')
            or name.startswith('mrr@') or name.startswith('judged_coverage@'))


def min_chunk_coverage(summary: dict) -> float:
    """Minimum chunk-level coverage over modes and over every reported cut-off (10 and 20)."""
    return min(value for levels in summary.values() for name, value in levels['chunk'].items()
               if name.startswith('min_judged_coverage@'))


def failed_top10_total(summary: dict) -> int:
    """Top-10 chunks whose judgment failed to parse, summed over modes at chunk level."""
    return sum(levels['chunk']['failed_top10_total'] for levels in summary.values())


COUNT_ROWS = {'cases', 'cases_without_relevant', 'cases_without_relevant_binary', 'failed_top10_total'}
# Graded rows that stay as reference when the headline uses grade >= 1.
REFERENCE_ROWS = ('ndcg@10', 'precision@10_grade1', 'precision@10_grade2')


def _cell(row: str, value) -> str:
    if value is None:
        return '—'
    return str(value) if row in COUNT_ROWS else f'{value:.3f}'


def render_markdown(summary: dict, *, judge: dict, pool: dict, threshold: int = 2) -> str:
    """``summary`` is ``{mode: {level: metrics}}``; ``judge`` has model, prompt_sha256, failed_parses;
    ``pool`` is pool.json; ``threshold`` is the grade the summary counted as relevant. At threshold 1
    the graded rows move to a "Reference (graded)" table under each level."""
    modes = list(summary)
    lines = [
        '# Graded retrieval comparison', '',
        f"- Judge: `{judge['model']}`, prompt `{judge['prompt_sha256'][:12]}`, failed parses: {judge['failed_parses']}",
        f"- Pool: corpus_version `{pool.get('corpus_version')}`, filters `{pool.get('filters')}`, "
        f"pool_depth {pool.get('pool_depth')}, settings_hash `{pool.get('settings_hash')}`",
        f'- Headline relevance: grade ≥ {threshold} (success@k, mrr, precision@10, ndcg@10_binary)',
        f"- Scoped to {pool.get('corpus_version')} with filter {pool.get('filters')}; "
        'multi-corpus fan-out is not measured.',
    ]

    def table(level: str, rows: list[str]) -> list[str]:
        return ['| Metric | ' + ' | '.join(modes) + ' |', '|---|' + '---:|' * len(modes)] + [
            f'| {row} | ' + ' | '.join(_cell(row, summary[m][level][row]) for m in modes) + ' |' for row in rows]
    for level in LEVELS:
        rows = list(summary[modes[0]][level])
        reference = [row for row in rows if row in REFERENCE_ROWS] if threshold == 1 else []
        lines += ['', f'## {level}', ''] + table(level, [row for row in rows if row not in reference])
        if reference:
            lines += ['', '### Reference (graded)', ''] + table(level, reference)
    if min_chunk_coverage(summary) < 1:
        lines += ['', '**WARNING:** some top-10 or top-20 chunks are unjudged; run judge_relevance.py first. '
                  'Metrics treat them as grade 0 and the comparison is not valid.']
    if failed_top10_total(summary):
        lines += ['', f'**Note:** {failed_top10_total(summary)} top-10 chunk slots (summed over modes) hold failed '
                  'judgments; they count as judged with gain 0. Rerun judge_relevance.py --retry-failed to grade them.']
    return '\n'.join(lines) + '\n'


def case_rows(per_case: dict, questions: dict[str, str], threshold: int = 2) -> tuple[list[str], list[dict]]:
    """Flatten ``per_case_metrics`` output to one row per case with ``{mode}.{level}.{metric}`` columns;
    ``relevance_threshold`` records the grade the metrics counted as relevant."""
    metrics = {mode: {level: [m for m in levels[level][0] if _is_case_metric(m)] if levels[level] else []
                      for level in LEVELS} for mode, levels in per_case.items()}
    fields = ['case_id', 'question', 'relevance_threshold'] + [f'{mode}.{level}.{metric}' for mode in per_case for level in LEVELS
                                        for metric in metrics[mode][level]]
    rows: dict[str, dict] = {}
    for mode, levels in per_case.items():
        for level in LEVELS:
            for result in levels[level]:
                row = rows.setdefault(result['case_id'], {'case_id': result['case_id'],
                                                          'question': questions.get(result['case_id'], ''),
                                                          'relevance_threshold': threshold})
                row.update({f'{mode}.{level}.{metric}': result[metric] for metric in metrics[mode][level]})
    return fields, sorted(rows.values(), key=lambda row: row['case_id'])


def write_cases_csv(path: Path, fields: list[str], rows: list[dict]) -> Path:
    with Path(path).open('x', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return Path(path)

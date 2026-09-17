#!/usr/bin/env python
"""Graded metrics for every recorded run in a pool directory.

Example:
    uv run python scripts/eval_graded.py --pool-dir data/eval/relevance/ip-v1-pilot-200 \
      --output data/eval/reports/graded/ip-v1-pilot-200
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

from legal_rag.evaluation.graded_metrics import per_case_metrics, summarize_runs
from legal_rag.evaluation.graded_report import (
    case_rows, failed_top10_total, min_chunk_coverage, render_markdown, write_cases_csv)
from legal_rag.evaluation.judgment_store import JudgeKey, JudgmentStore
from legal_rag.evaluation.relevance_judge import PROMPT_SHA256
from legal_rag.evaluation.relevance_pool import read_jsonl

app = typer.Typer(add_completion=False)


@app.command()
def main(
    pool_dir: Path = typer.Option(..., exists=True, file_okay=False),
    model: str = typer.Option('qwen3:8b'),
    output: Path = typer.Option(None, help='New directory for graded.json, graded.md and cases.csv.'),
    allow_partial: bool = typer.Option(False, help='Write --output even if some top-10/top-20 chunks are unjudged.'),
) -> None:
    key = JudgeKey(f'ollama:{model}', PROMPT_SHA256)
    try:
        store = JudgmentStore(pool_dir / 'judgments.jsonl')
        grades, documents, failures = store.grades(key), store.documents(key), set(store.failures(key))
        runs = {path.stem: read_jsonl(path) for path in sorted((pool_dir / 'runs').glob('*.jsonl'))}
        if not runs:
            raise ValueError(f'no runs in {pool_dir / "runs"}')
        pool = json.loads((pool_dir / 'pool.json').read_text(encoding='utf-8'))
        depth = int(pool.get('pool_depth', 20))
        summary = summarize_runs(runs, grades, documents, failures, depth)
    except (ValueError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    judge = {'model': key.model, 'prompt_sha256': key.prompt_sha256, 'failed_parses': len(store.failures(key))}
    text = render_markdown(summary, judge=judge, pool=pool)
    typer.echo(text)
    if failed_top10_total(summary):
        typer.echo(f'WARNING: {failed_top10_total(summary)} top-10 chunk slots (summed over modes) hold failed '
                   'judgments, counted as grade 0; consider judge_relevance.py --retry-failed', err=True)
    if not output:
        return
    if min_chunk_coverage(summary) < 1 and not allow_partial:
        typer.echo('refusing to write --output with unjudged top-10/top-20 chunks; judge them or pass --allow-partial',
                   err=True)
        raise typer.Exit(1)
    questions = {row['case_id']: row['question'] for row in read_jsonl(pool_dir / 'pool.jsonl')}
    fields, rows = case_rows(per_case_metrics(runs, grades, documents, failures, depth), questions)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        typer.echo(f'{output} already exists', err=True)
        raise typer.Exit(1) from exc
    report = {'judge': judge, 'pool': pool, 'modes': summary}
    (output / 'graded.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'graded.md').write_text(text, encoding='utf-8')
    write_cases_csv(output / 'cases.csv', fields, rows)
    typer.echo(f'wrote {output}', err=True)


if __name__ == '__main__':
    app()

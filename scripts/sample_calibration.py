#!/usr/bin/env python
"""Draw the stratified 50-pair human calibration set from a relevance pool.

Example:
    uv run python scripts/sample_calibration.py --pool-dir data/eval/relevance/ip-v1-pilot-200
"""
from __future__ import annotations

from pathlib import Path

import typer

from legal_rag.evaluation.calibration import sample_calibration, write_calibration
from legal_rag.evaluation.relevance_pool import read_jsonl

app = typer.Typer(add_completion=False)


@app.command()
def main(
    pool_dir: Path = typer.Option(..., exists=True, file_okay=False, help='Directory with pool.jsonl.'),
    seed: int = typer.Option(42),
) -> None:
    try:
        sample = sample_calibration(read_jsonl(pool_dir / 'pool.jsonl'), seed=seed)
        write_calibration(pool_dir, sample)
    except (ValueError, OSError) as exc:
        typer.echo(f'{type(exc).__name__}: {exc}', err=True)
        raise typer.Exit(1) from exc
    typer.echo(f'wrote {pool_dir / "calibration.csv"}: {len(sample)} pairs; fill human_grade (0/1/2) '
               'before running judge_relevance.py')


if __name__ == '__main__':
    app()

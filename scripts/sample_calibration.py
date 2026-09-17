#!/usr/bin/env python
"""Draw the stratified calibration set from a relevance pool.

Example:
    uv run python scripts/sample_calibration.py --pool-dir data/eval/relevance/ip-v1-pilot-200

A depth-10 pool derived with derive_relevance_pool.py, excluding the first calibration's pairs:
    uv run python scripts/sample_calibration.py --pool-dir data/eval/relevance/ip-v1-pilot-200-top10 \
      --seed 43 --quota rank_1_3=25 --quota rank_4_10=25 --exclude-calibration data/eval/relevance/ip-v1-pilot-200
"""
from __future__ import annotations

from pathlib import Path

import typer

from legal_rag.evaluation.calibration import (
    STRATA, calibration_pair_ids, parse_quotas, sample_calibration, write_calibration)
from legal_rag.evaluation.relevance_pool import read_jsonl

app = typer.Typer(add_completion=False)


@app.command()
def main(
    pool_dir: Path = typer.Option(..., exists=True, file_okay=False, help='Directory with pool.jsonl.'),
    seed: int = typer.Option(42),
    quota: list[str] = typer.Option(
        [], help=f'STRATUM=N, repeatable, drawn in the given order (default {STRATA}). Without a gold quota, '
                 'gold chunks are placed by rank.'),
    exclude_calibration: list[Path] = typer.Option(
        [], exists=True, file_okay=False, help="Pool directory whose calibration.json pairs are never drawn; repeatable."),
) -> None:
    try:
        quotas = parse_quotas(quota)
        exclude = set().union(*(calibration_pair_ids(path) for path in exclude_calibration))
        sample = sample_calibration(read_jsonl(pool_dir / 'pool.jsonl'), seed=seed, quotas=quotas, exclude=exclude)
        write_calibration(pool_dir, sample)
    except (ValueError, OSError, KeyError) as exc:
        typer.echo(f'{type(exc).__name__}: {exc}', err=True)
        raise typer.Exit(1) from exc
    typer.echo(f'wrote {pool_dir / "calibration.csv"}: {len(sample)} pairs {quotas}, seed {seed}, '
               f'excluded {len(exclude)} earlier calibration pairs; fill codex_grade (0/1/2) '
               'before running judge_relevance.py')


if __name__ == '__main__':
    app()

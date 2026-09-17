#!/usr/bin/env python
"""Derive a shallower relevance pool from an existing one, without retrieving again.

Keeps pairs ranked within --pool-depth by any mode, copies runs/ unchanged, and copies
judgments.jsonl and judge_runs.jsonl so pairs already judged are reused. The source is only read.

Example:
    uv run python scripts/derive_relevance_pool.py --from data/eval/relevance/ip-v1-pilot-200 \
      --output data/eval/relevance/ip-v1-pilot-200-top10 --pool-depth 10
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

from legal_rag.evaluation.relevance_pool import derive_pool_dir

app = typer.Typer(add_completion=False)


@app.command()
def main(
    source: Path = typer.Option(..., '--from', exists=True, file_okay=False, help='Existing pool directory.'),
    output: Path = typer.Option(..., help='New directory; must not exist.'),
    pool_depth: int = typer.Option(..., min=1, help='At most the source pool_depth.'),
    include_unretrieved_gold: bool = typer.Option(
        False, '--include-unretrieved-gold/--no-include-unretrieved-gold',
        help='Keep gold chunks that no mode retrieved within --pool-depth.'),
) -> None:
    try:
        derived = derive_pool_dir(source, output, pool_depth=pool_depth,
                                  include_unretrieved_gold=include_unretrieved_gold)
    except (ValueError, OSError) as exc:
        typer.echo(f'{type(exc).__name__}: {exc}', err=True)
        raise typer.Exit(1) from exc
    meta = json.loads((derived / 'pool.json').read_text(encoding='utf-8'))
    typer.echo(f"wrote {derived}: {meta['pairs']} pairs at pool_depth {meta['pool_depth']}; "
               f"judgments copied from {meta['judgments_copied_from']}")


if __name__ == '__main__':
    app()

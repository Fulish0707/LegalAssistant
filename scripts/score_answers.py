#!/usr/bin/env python
"""Score a filled pairwise.csv against the hidden side mapping.

Example:
    uv run python scripts/score_answers.py --comparison-dir data/eval/answers/v1-vs-v2 --side ip-v2
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import typer

from legal_rag.evaluation.answers.answer_compare import win_rate

app = typer.Typer(add_completion=False)


@app.command()
def main(comparison_dir: Path = typer.Option(..., exists=True, file_okay=False),
         side: str = typer.Option(..., help='The side whose win rate is reported.')) -> None:
    manifest = json.loads((comparison_dir / 'pairwise_key.json').read_text(encoding='utf-8'))
    if side not in manifest['sides']:
        raise typer.BadParameter(f"unknown side {side!r}; choose from {list(manifest['sides'])}")
    with (comparison_dir / 'pairwise.csv').open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    verdicts = {row['case_id']: (row.get('winner') or '').strip() for row in rows}
    blank = [case for case, winner in verdicts.items() if not winner]
    if blank:
        typer.echo(f'{len(blank)} of {len(rows)} pairs have no winner; judge them first', err=True)
        raise typer.Exit(1)
    try:
        result = win_rate(verdicts, manifest['key'], side=side)
    except (ValueError, KeyError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    low, high = result['ci']
    typer.echo(json.dumps({**result, 'model': manifest['model'], 'k': manifest['k']}, ensure_ascii=False, indent=2))
    verdict = 'decisive' if result['decisive'] else 'not decisive: the interval spans 50%'
    typer.echo(f"{side} wins {result['win_rate']:.1%} of pairs (95% CI {low:.1%}-{high:.1%}) — {verdict}")


if __name__ == '__main__':
    app()

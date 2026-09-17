#!/usr/bin/env python
"""Combine finished diagnose_retrieval.py runs (dense / sparse / hybrid) into one report.

Example:
    uv run python scripts/compare_retrieval_modes.py \
      data/eval/reports/retrieval_diagnostics/ip-v1/intellectual_property/{dense,sparse,hybrid}-seed42-final \
      --output data/eval/reports/retrieval_diagnostics/ip-v1/intellectual_property/comparison-seed42-final
"""
from __future__ import annotations

from pathlib import Path

import typer

from legal_rag.evaluation.mode_comparison import compare_mode_runs, write_mode_comparison

app = typer.Typer(add_completion=False)


@app.command()
def main(
    runs: list[Path] = typer.Argument(..., exists=True, file_okay=False, help='Run directories, one per mode.'),
    output: Path = typer.Option(..., help='New directory; an existing one is refused.'),
) -> None:
    try:
        report = compare_mode_runs(runs)
        write_mode_comparison(output, report)
    except (ValueError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f'wrote {output}')


if __name__ == '__main__':
    app()

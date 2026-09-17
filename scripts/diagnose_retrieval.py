#!/usr/bin/env python
"""Inspect retrieval failures without changing indexing or retrieval algorithms."""
from __future__ import annotations

from pathlib import Path

import typer

from legal_rag.cli import settings_from_options
from legal_rag.evaluation.diagnostic_runner import run_retrieval_diagnostics

app = typer.Typer(add_completion=False)


@app.command()
def main(
    domain: str = typer.Option(..., help='QA and retrieval legal domain.'),
    profile: str = typer.Option(None),
    root: Path = typer.Option(None),
    corpus_version: str = typer.Option(None),
    retrieval_mode: str = typer.Option('hybrid', help='dense | sparse | hybrid'),
    limit: int = typer.Option(50, min=1, help='Maximum review cases; evaluated AFTER searching the target set.'),
    top_k: int = typer.Option(20, min=1),
    deep_k: int = typer.Option(100, min=1),
    seed: int = typer.Option(42),
    only_failures: bool = typer.Option(False, help='Select only failures at Top-K for review.'),
    output: Path = typer.Option(None, help='New run directory; existing directories are never overwritten.'),
    qa_file: Path = typer.Option(None, exists=True, dir_okay=False, help='Processed QA; defaults to corpus/source manifest QA.'),
    dataset: Path = typer.Option(None, exists=True, dir_okay=False, help='Explicit evaluation JSONL; defaults to eval/<corpus_version>.jsonl if present.'),
    preview_chars: int = typer.Option(1200, min=1),
    max_gold_chunks: int = typer.Option(10, min=1),
) -> None:
    settings = settings_from_options(profile=profile, root=root, corpus_version=corpus_version)
    try:
        destination = run_retrieval_diagnostics(
            settings, domain=domain, retrieval_mode=retrieval_mode, limit=limit,
            top_k=top_k, deep_k=deep_k, seed=seed, only_failures=only_failures,
            output=output, qa_file=qa_file, dataset=dataset,
            preview_chars=preview_chars, max_gold_chunks=max_gold_chunks,
            progress=lambda message: typer.echo(message, err=True),
        )
    except (ValueError, OSError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f'wrote {destination}')


if __name__ == '__main__':
    app()

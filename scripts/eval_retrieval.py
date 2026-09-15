#!/usr/bin/env python
"""Retrieval metrics (Recall@k, Precision@k, MRR, nDCG@k) over a JSONL dataset.

Example:
    uv run python scripts/eval_retrieval.py --dataset data/eval/retrieval.jsonl --json out.json
"""

from __future__ import annotations

from pathlib import Path

import typer

from legal_rag.cli import settings_from_options
from legal_rag.evaluation import load_retrieval_examples, run_retrieval_eval
from legal_rag.factory import build_hybrid_retriever, select_legs

app = typer.Typer(add_completion=False)


@app.command()
def main(
    dataset: Path = typer.Option(..., exists=True),
    profile: str = typer.Option(None),
    root: Path = typer.Option(None),
    corpus_version: str = typer.Option(None),
    embedding_backend: str = typer.Option(None),
    leg: str = typer.Option("hybrid", help="hybrid | dense | sparse"),
    json_out: Path = typer.Option(None, "--json"),
) -> None:
    extra = {"embedding": {"backend": embedding_backend}} if embedding_backend else None
    settings = settings_from_options(profile=profile, root=root, corpus_version=corpus_version, extra=extra)
    hybrid = build_hybrid_retriever(settings)
    retriever = select_legs(hybrid, leg)
    report = run_retrieval_eval(retriever, load_retrieval_examples(dataset))
    typer.echo(report.table())
    if json_out:
        report.write(json_out)
        typer.echo(f"wrote {json_out}")


if __name__ == "__main__":
    app()

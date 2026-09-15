#!/usr/bin/env python
"""Build a retrieval evaluation dataset from an external qa.jsonl.

Only the question and its document reference are carried over; the gold answer stays
out of the dataset entirely. Questions whose document is absent from the corpus are
dropped, because they can never be retrieved and would depress Recall@k for a
data-matching reason rather than a retrieval one.

Example:
    uv run python scripts/build_eval_set.py \
      --qa-file /home/layke/Documents/legal-rag/data/processed/criminal/qa.jsonl \
      --corpus-version criminal-v1 --max-examples 2000
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from legal_rag.cli import settings_from_options
from legal_rag.evaluation.qa_dataset import build_retrieval_examples, corpus_document_ids, write_retrieval_examples

app = typer.Typer(add_completion=False)


@app.command()
def main(
    qa_file: Path = typer.Option(..., exists=True, dir_okay=False, help="External qa.jsonl; never indexed."),
    corpus_version: str = typer.Option(..., help="Corpus whose documents define what is retrievable."),
    output: Path = typer.Option(None, help="Defaults to <eval_dir>/<corpus_version>.jsonl"),
    profile: str = typer.Option(None),
    root: Path = typer.Option(None),
    with_filters: bool = typer.Option(False, help="Attach legal_domain/document_type filters to each example."),
    max_examples: int = typer.Option(None, min=1, help="Deterministically sample at most this many questions."),
    seed: int = typer.Option(0, help="Sampling seed; the same seed yields the same subset."),
) -> None:
    settings = settings_from_options(profile=profile, root=root, corpus_version=corpus_version)
    corpus_dir = settings.processed_corpus_dir()
    if not corpus_dir.exists():
        typer.echo(f"no corpus at {corpus_dir}; import it first", err=True)
        raise typer.Exit(1)
    destination = output or settings.resolve(settings.paths.eval_dir) / f"{corpus_version}.jsonl"
    try:
        known = corpus_document_ids(corpus_dir)
        examples, report = build_retrieval_examples(
            qa_file, known, with_filters=with_filters, max_examples=max_examples, seed=seed
        )
        written = write_retrieval_examples(examples, destination)
    except (ValueError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps({**report.summary(), "corpus_documents": len(known),
                           "written": written, "output": str(destination)},
                          ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()

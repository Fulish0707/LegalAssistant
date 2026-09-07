#!/usr/bin/env python
"""Normalize raw legal data into a reproducible processed corpus.

Example:
    uv run python scripts/ingest.py --raw-dir data/raw --corpus-version v1
"""

from __future__ import annotations

from pathlib import Path

import typer

from legal_rag.cli import settings_from_options
from legal_rag.ingestion.pipeline import run_ingestion

app = typer.Typer(add_completion=False)


@app.command()
def main(
    profile: str = typer.Option(None, help="Config profile name under configs/ (default: LEGAL_RAG_PROFILE or local)."),
    root: Path = typer.Option(None, help="Project root that relative data paths resolve against."),
    raw_dir: Path = typer.Option(None, help="Directory holding immutable raw datasets."),
    corpus_version: str = typer.Option(None, help="Label for the processed corpus, e.g. v1."),
) -> None:
    settings = settings_from_options(profile=profile, root=root, raw_dir=raw_dir, corpus_version=corpus_version)
    report = run_ingestion(settings)
    typer.echo(
        f"documents={report.documents} chunks={report.chunks} deduplicated={report.deduplicated} "
        f"unrecognized={report.unrecognized} output={report.output_dir}"
    )


if __name__ == "__main__":
    app()

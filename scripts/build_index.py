#!/usr/bin/env python
"""Embed and index a processed corpus into Qdrant.

Example:
    uv run python scripts/build_index.py --corpus-version v1
"""

from __future__ import annotations

from pathlib import Path

import typer

from legal_rag.cli import settings_from_options
from legal_rag.retrieval.index_builder import build_index

app = typer.Typer(add_completion=False)


@app.command()
def main(
    profile: str = typer.Option(None),
    root: Path = typer.Option(None),
    corpus_version: str = typer.Option(None),
    embedding_backend: str = typer.Option(None, help="ollama | fake"),
) -> None:
    extra = {"embedding": {"backend": embedding_backend}} if embedding_backend else None
    settings = settings_from_options(profile=profile, root=root, corpus_version=corpus_version, extra=extra)
    report = build_index(settings, progress=lambda done, total: typer.echo(f"  indexed {done}/{total}", err=True))
    typer.echo(f"chunks={report.chunks} collection={report.collection} dimension={report.dimension} index_dir={report.index_dir}")


if __name__ == "__main__":
    app()

#!/usr/bin/env python
"""Embed and index a processed corpus into Qdrant.

Example:
    uv run python scripts/build_index.py --corpus-version v1
"""

from __future__ import annotations

import time
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
    resume: bool = typer.Option(False, "--resume", help="Preserve the collection and embed only missing chunks."),
    on_disk: bool = typer.Option(False, help="Memory-map vectors and the HNSW graph; for a corpus larger than the server's RAM."),
) -> None:
    extra = {"embedding": {"backend": embedding_backend}} if embedding_backend else None
    settings = settings_from_options(profile=profile, root=root, corpus_version=corpus_version, extra=extra)
    if resume:
        typer.echo("Resuming: checking BM25 statistics and stored chunks; existing points are preserved.", err=True)
    started = time.monotonic()

    def progress(done: int, total: int) -> None:
        elapsed = time.monotonic() - started
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = f", about {(total - done) / rate / 60:.0f} min left" if rate and done < total else ""
        typer.echo(f"  indexed {done}/{total} ({rate:.1f} chunks/s{eta})", err=True)

    report = build_index(settings, on_disk=on_disk, resume=resume, progress=progress)
    typer.echo(f"chunks={report.chunks} collection={report.collection} dimension={report.dimension} index_dir={report.index_dir}")


if __name__ == "__main__":
    app()

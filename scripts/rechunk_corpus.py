#!/usr/bin/env python
"""Re-cut a processed corpus along its own section structure.

Documents are copied unchanged; only the chunks are re-derived. The output version is
new: an existing directory is never overwritten.

Example:
    uv run python scripts/rechunk_corpus.py --corpus-version ip-v1 --output-version ip-v2
"""

from __future__ import annotations

from pathlib import Path

import typer

from legal_rag.chunking.imported import ChunkingOptions
from legal_rag.cli import settings_from_options
from legal_rag.ingestion.rechunk import rechunk_corpus

app = typer.Typer(add_completion=False)


def _table(title: str, rows: dict) -> None:
    typer.echo(f"{title}:")
    for key, value in rows.items():
        typer.echo(f"  {key}: {value}")


@app.command()
def main(
    corpus_version: str = typer.Option(..., help="Processed corpus to read documents from."),
    output_version: str = typer.Option(..., help="New corpus version to write."),
    profile: str = typer.Option(None),
    root: Path = typer.Option(None),
    limit: int = typer.Option(None, min=1, help="Process only the first N documents."),
    target_chars: int = typer.Option(1200, min=1),
    max_chars: int = typer.Option(2000, min=1),
    merge_threshold: int = typer.Option(400, min=1),
    examples: int = typer.Option(3, min=0, help="Example chunk previews kept per document type."),
) -> None:
    settings = settings_from_options(profile=profile, root=root, corpus_version=corpus_version)
    source = settings.processed_corpus_dir()
    destination = source.parent / output_version
    options = ChunkingOptions(target_chars=target_chars, max_chars=max_chars, merge_threshold=merge_threshold)
    typer.echo(f"reading {source}\nwriting {destination}", err=True)
    report = rechunk_corpus(source, destination, corpus_version=output_version, options=options,
                            limit=limit, examples=examples)
    typer.echo(f"documents={report['documents']} chunks={report['chunks']} "
               f"({report['chunks_per_document']} per document)")
    _table("parsed structurally", report["parsed"])
    _table("fell back to windows", report["fallback"])
    _table("chunk characters", report["chunk_chars"])
    typer.echo(f"chunks opening with a header line: {report['header_line_share']:.1%}")
    typer.echo(f"chunks headed by a real section path: {report['structural_heading_share']:.1%}")
    typer.echo(f"full report: {destination / 'report.json'}")


if __name__ == "__main__":
    app()

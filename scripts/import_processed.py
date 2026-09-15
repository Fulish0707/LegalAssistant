#!/usr/bin/env python
"""Import externally preprocessed documents.jsonl and chunks.jsonl into a new corpus."""
from pathlib import Path
import json
import typer

from legal_rag.cli import settings_from_options
from legal_rag.ingestion.import_processed import import_processed

app = typer.Typer(add_completion=False)


@app.command()
def main(
    input_dir: Path = typer.Option(..., exists=True, file_okay=False),
    corpus_version: str = typer.Option(..., help='New destination corpus version; existing output is never overwritten.'),
    profile: str = typer.Option(None),
    root: Path = typer.Option(None),
    max_documents: int = typer.Option(None, min=1, help='Import all chunks of the first N documents; validate all input.'),
    validate_only: bool = typer.Option(False, help='Validate and print report without publishing a corpus.'),
) -> None:
    settings = settings_from_options(profile=profile, root=root, corpus_version=corpus_version)
    try:
        report = import_processed(input_dir, settings, max_documents=max_documents, validate_only=validate_only)
    except (ValueError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    app()

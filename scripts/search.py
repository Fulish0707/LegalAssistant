#!/usr/bin/env python
"""Hybrid search over an index, for debugging retrieval.

Example:
    uv run python scripts/search.py "민법 제750조" --k 5
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from legal_rag.cli import settings_from_options
from legal_rag.factory import build_hybrid_retriever
from legal_rag.schemas import DocumentType, RetrievalFilters

app = typer.Typer(add_completion=False)


@app.command()
def main(
    query: str = typer.Argument(...),
    k: int = typer.Option(5),
    document_type: str = typer.Option(None, help="statute | court_decision | admin_interpretation"),
    profile: str = typer.Option(None),
    root: Path = typer.Option(None),
    corpus_version: str = typer.Option(None),
    embedding_backend: str = typer.Option(None),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    extra = {"embedding": {"backend": embedding_backend}} if embedding_backend else None
    settings = settings_from_options(profile=profile, root=root, corpus_version=corpus_version, extra=extra)
    retriever = build_hybrid_retriever(settings)
    filters = RetrievalFilters(document_types=[DocumentType(document_type)]) if document_type else None
    hits = retriever.retrieve(query, k, filters)
    if as_json:
        typer.echo(json.dumps([hit.model_dump(mode="json") for hit in hits], ensure_ascii=False, indent=2))
        return
    for rank, hit in enumerate(hits, start=1):
        header = hit.chunk.text.split("\n", 1)[0]
        typer.echo(f"{rank:>2}. {hit.score:.4f}  {hit.chunk.id}")
        typer.echo(f"     {header}")
    for leg, rows in retriever.last_trace.items():
        if leg not in ("fused", "reranked"):
            typer.echo(f"[{leg}] {[cid.split('#')[1] for cid, _ in rows[:3]]}", err=True)


if __name__ == "__main__":
    app()

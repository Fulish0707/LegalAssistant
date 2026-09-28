#!/usr/bin/env python
"""Write the derived opening article into already indexed statute windows.

`Corpus.chunks()` names the article an imported statute window opens in (see
`legal_rag.ingestion.statute_articles`), so a fresh index gets it. This copies the same
values into existing Qdrant payloads with `set_payload`, without re-embedding. Retrieval
does not read `metadata.article`; only the evidence header and the verifier do.

Example (dry run, then write):
    uv run python scripts/backfill_statute_articles.py --profile local
    uv run python scripts/backfill_statute_articles.py --profile local --apply
"""
from __future__ import annotations

from collections import defaultdict

import typer

from legal_rag.cli import settings_from_options
from legal_rag.factory import build_embedder, collection_name, open_store
from legal_rag.retrieval.store import point_id
from legal_rag.ingestion.corpus import Corpus
from legal_rag.schemas import DocumentType

app = typer.Typer(add_completion=False)


@app.command()
def main(
    profile: str = typer.Option(None),
    corpus_version: list[str] = typer.Option(None, help="Corpora to update; default: the profile's served corpora."),
    apply: bool = typer.Option(False, help="Write the payloads; without it only counts are printed."),
) -> None:
    base = settings_from_options(profile=profile)
    for version in corpus_version or base.retrieval.corpora or [base.corpus.version]:
        settings = settings_from_options(profile=profile, corpus_version=version)
        embedder = build_embedder(settings)
        store = open_store(settings, embedder)
        by_article: dict[str, list[str]] = defaultdict(list)
        statutes = 0
        for chunk in Corpus.load(settings.processed_corpus_dir()).chunks():
            if chunk.document_type is DocumentType.statute:
                statutes += 1
                if chunk.metadata.article:
                    by_article[chunk.metadata.article].append(chunk.id)
        named = sum(map(len, by_article.values()))
        typer.echo(f"{version}: {named}/{statutes} statute chunks named, collection {collection_name(settings, embedder)}")
        if apply:
            for article, chunk_ids in by_article.items():
                store.client.set_payload(store.collection, payload={"article": article}, key="metadata",
                                         points=[point_id(i) for i in chunk_ids], wait=True)
            typer.echo("  written")
        store.close()


if __name__ == "__main__":
    app()

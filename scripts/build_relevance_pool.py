#!/usr/bin/env python
"""Retrieve a diagnostic target in several modes and pool chunks for relevance judging.

Example:
    uv run python scripts/build_relevance_pool.py --profile local --corpus-version ip-v1 \
      --domain intellectual_property \
      --target data/eval/reports/retrieval_diagnostics/ip-v1/intellectual_property/hybrid-seed42-final/target.jsonl \
      --output data/eval/relevance/ip-v1-pilot-200
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import typer

from legal_rag.cli import settings_from_options
from legal_rag.evaluation.diagnosis.diagnostic_runner import _gold_chunk_index, _qa_path, deepen_retrieval, load_diagnostic_rows
from legal_rag.evaluation.relevance.relevance_pool import build_pool, read_jsonl, write_pool
from legal_rag.factory import build_hybrid_retriever, close_legs, select_legs
from legal_rag.ingestion.corpus import Corpus

app = typer.Typer(add_completion=False)


def _progress(message: str) -> None:
    typer.echo(message, err=True)


@app.command()
def main(
    domain: str = typer.Option(...),
    target: Path = typer.Option(..., exists=True, dir_okay=False, help='target.jsonl from a diagnostic run.'),
    output: Path = typer.Option(..., help='New directory; existing directories are never overwritten.'),
    profile: str = typer.Option(None),
    root: Path = typer.Option(None),
    corpus_version: str = typer.Option(None),
    qa_file: Path = typer.Option(None, exists=True, dir_okay=False),
    modes: list[str] = typer.Option(['dense', 'sparse', 'hybrid'], '--mode'),
    pool_depth: int = typer.Option(20, min=1),
    run_depth: int = typer.Option(100, min=1),
    max_gold_chunks: int = typer.Option(10, min=1),
) -> None:
    if output.exists():
        raise typer.BadParameter(f'{output} exists; choose a new --output')
    settings = deepen_retrieval(settings_from_options(profile=profile, root=root, corpus_version=corpus_version),
                                run_depth)
    _progress(f'settings hash {settings.hash()}')
    qa_file = _qa_path(settings, qa_file)
    cases, _, provenance = load_diagnostic_rows(qa_file, domain=domain, dataset=target)
    target_ids = {row['id'] for row in read_jsonl(target)}
    if {case['case_id'] for case in cases} != target_ids:
        raise typer.BadParameter('QA rows resolved from --target do not reproduce its case ids')
    _progress(f'{len(cases)} cases; scanning gold chunks')
    # Previews are not truncated: the pool keeps full gold chunk text for judging.
    gold, _ = _gold_chunk_index(Corpus.load(settings.processed_corpus_dir()), {c['document_id'] for c in cases},
                                max_gold_chunks=max_gold_chunks, preview_chars=10**9)
    hybrid = build_hybrid_retriever(settings)
    try:
        retrievers = {mode: select_legs(hybrid, mode) for mode in modes}
        pool, runs = build_pool(cases, retrievers, gold_chunks=gold, pool_depth=pool_depth, run_depth=run_depth,
                                progress=_progress)
    finally:
        close_legs(hybrid)
    write_pool(output, pool, runs, {
        'domain': domain,
        'target': str(target.resolve()), 'target_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'qa_file': provenance['qa_file'], 'corpus_version': settings.corpus.version,
        'retrieval_corpora': settings.retrieval.corpora or [settings.corpus.version],
        'filters': {'legal_domain': domain}, 'pool_depth': pool_depth, 'run_depth': run_depth,
        'reranker': settings.rerank.backend, 'settings_hash': settings.hash()})
    typer.echo(f'wrote {output}: {len(pool)} pairs')


if __name__ == '__main__':
    app()

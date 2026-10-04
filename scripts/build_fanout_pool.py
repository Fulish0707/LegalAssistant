#!/usr/bin/env python
"""Retrieve a target under several corpus fan-outs and pool the hits for relevance judging.

Unlike build_relevance_pool.py, retrieval here is unfiltered by default and a run may span any
set of corpora, including each case's own one (``own``), so the cost of fanning out and the
benefit of not routing can both be measured. The configured reranker runs inside every retrieval,
so each recorded run is a reranked ranking, not a fused one.

Example:
    uv run python scripts/build_fanout_pool.py --profile local --qa-root ~/Documents/legal-rag/data/processed \
      --case-source intellectual_property=ip-v1=data/eval/.../target.jsonl \
      --run ip_only=ip-v1 --run fanout=criminal-v1,civil-v1,administrative-v1,ip-v1 \
      --copy-judgments data/eval/relevance/ip-v1-pilot-200-rerank-k50 \
      --output data/eval/relevance/ip-200-fanout
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import typer

from legal_rag.cli import settings_from_options
from legal_rag.evaluation.diagnostic_runner import gold_chunks_across_corpora, load_diagnostic_rows
from legal_rag.evaluation.fusion_sweep import pending_pairs
from legal_rag.evaluation.judgment_store import JudgmentStore
from legal_rag.evaluation.relevance_judge import JUDGE_THINK, describe_key, judge_key
from legal_rag.evaluation.relevance_pool import (
    JUDGMENT_FILES, adopt_run, build_pool, domain_filters, no_filters, read_jsonl, write_pool)
from legal_rag.factory import build_embedder, build_hybrid_retriever, build_reranker, close_legs, corpus_settings
from legal_rag.ingestion.corpus import Corpus

app = typer.Typer(add_completion=False)
CALIBRATION_FILES = ('calibration.csv', 'calibration.json')
SECONDS_PER_PAIR = 15.3
OWN = 'own'


def _progress(message: str) -> None:
    typer.echo(message, err=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _split(entry: str, option: str, fields: int) -> list[str]:
    parts = entry.split('=')
    if len(parts) != fields or not all(parts):
        raise typer.BadParameter(f'{option} {entry!r}: expected {fields} nonempty parts separated by "="')
    return parts


@app.command()
def main(
    case_source: list[str] = typer.Option(..., '--case-source',
                                          help='Repeatable DOMAIN=CORPUS=TARGET_JSONL.'),
    run: list[str] = typer.Option(..., '--run',
                                  help=f'Repeatable NAME=CORPORA, comma separated, or NAME={OWN} '
                                       "for each case's own corpus."),
    output: Path = typer.Option(..., help='New directory; existing directories are never overwritten.'),
    qa_root: Path = typer.Option(..., exists=True, file_okay=False, help='Directory holding <domain>/qa.jsonl.'),
    adopt: list[str] = typer.Option([], '--adopt', help='Repeatable NAME=POOL_DIR=RUN, to record a run '
                                                        'retrieved elsewhere over the same cases.'),
    copy_judgments: Path = typer.Option(None, exists=True, file_okay=False,
                                        help='Pool directory whose judgments are reused.'),
    copy_calibration: bool = typer.Option(False, help='Also copy calibration.csv/json from --copy-judgments; '
                                                      'only sound when its pairs come from these cases.'),
    domain_filter: bool = typer.Option(False, '--domain-filter/--no-domain-filter',
                                       help='Retrieve each case inside its own legal domain, as earlier pools did.'),
    pool_depth: int = typer.Option(10, min=1),
    run_depth: int = typer.Option(50, min=1),
    max_gold_chunks: int = typer.Option(10, min=1),
    judge_model: str = typer.Option('qwen3:8b', help='Judge whose existing judgments count as done.'),
    think: bool = typer.Option(JUDGE_THINK, '--think/--no-think'),
    prompt_profile: str = typer.Option('corpus', help='Relevance instructions: corpus, shared or corpus-experimental.'),
    judge_domain: str = typer.Option(None, help='Use the same domain override as the judging command.'),
    profile: str = typer.Option(None),
    root: Path = typer.Option(None),
) -> None:
    if output.exists():
        raise typer.BadParameter(f'{output} exists; choose a new --output')
    settings = settings_from_options(profile=profile, root=root)
    if settings.retrieval.k_fused != run_depth:
        raise typer.BadParameter(f'--run-depth {run_depth} but the configuration fuses {settings.retrieval.k_fused}; '
                                 'a run can only record what the served pipeline returns')
    sources = [_split(entry, '--case-source', 3) for entry in case_source]
    runs_wanted = {name: (OWN if spec == OWN else tuple(spec.split(',')))
                   for name, spec in (_split(entry, '--run', 2) for entry in run)}

    cases, corpus_of, targets = [], {}, {}
    for domain, version, target in sources:
        rows, _, provenance = load_diagnostic_rows(qa_root / domain / 'qa.jsonl', domain=domain, dataset=Path(target))
        ids = {row['id'] for row in read_jsonl(Path(target))}
        if {row['case_id'] for row in rows} != ids:
            raise typer.BadParameter(f'{target}: QA rows do not reproduce its case ids')
        cases += rows
        corpus_of[domain] = version
        targets[domain] = {'target': str(Path(target).resolve()), 'target_sha256': _sha256(Path(target)),
                           'corpus_version': version, 'qa_file': provenance['qa_file'], 'cases': len(rows)}
        _progress(f'{domain}: {len(rows)} cases from {target}')
    cases.sort(key=lambda case: case['case_id'])

    _progress(f'scanning {len(corpus_of)} corpora for the gold chunks of {len(cases)} cases')
    gold = gold_chunks_across_corpora(
        {version: Corpus.load(corpus_settings(settings, version).processed_corpus_dir())
         for version in dict.fromkeys(corpus_of.values())},
        {case['document_id'] for case in cases}, max_gold_chunks=max_gold_chunks, preview_chars=10 ** 9)

    embedder, reranker = build_embedder(settings), build_reranker(settings)
    needed = {spec for spec in runs_wanted.values() if spec != OWN}
    if OWN in runs_wanted.values():
        needed |= {(version,) for version in corpus_of.values()}
    retrievers = {}
    started = time.perf_counter()
    try:
        for versions in sorted(needed):
            scoped = settings.model_copy(update={
                'retrieval': settings.retrieval.model_copy(update={'corpora': list(versions)})})
            retrievers[versions] = build_hybrid_retriever(scoped, embedder=embedder, reranker=reranker)
            _progress(f'retriever over {", ".join(versions)}')
        modes = {name: (retrievers[spec] if spec != OWN else
                        (lambda case, table=corpus_of: retrievers[(table[case['legal_domain']],)]))
                 for name, spec in runs_wanted.items()}
        pool, runs = build_pool(cases, modes, gold_chunks=gold, pool_depth=pool_depth, run_depth=run_depth,
                                progress=_progress, case_filters=domain_filters if domain_filter else no_filters)
    finally:
        for retriever in retrievers.values():
            close_legs(retriever)
    elapsed = time.perf_counter() - started
    _progress(f'retrieved in {elapsed / 60:.1f} min '
              f'({elapsed / max(len(cases) * len(runs_wanted), 1):.2f} s/query)')

    adopted = {}
    for name, source_dir, source_run in (_split(entry, '--adopt', 3) for entry in adopt):
        pool, runs = adopt_run(pool, runs, name, read_jsonl(Path(source_dir) / 'runs' / f'{source_run}.jsonl'),
                               read_jsonl(Path(source_dir) / 'pool.jsonl'), pool_depth=pool_depth)
        adopted[name] = {'pool_dir': str(source_dir), 'run': source_run,
                         'run_sha256': _sha256(Path(source_dir) / 'runs' / f'{source_run}.jsonl')}
        _progress(f'adopted {source_run} from {source_dir} as {name}')

    copied = []
    if copy_judgments:
        copied = [name for name in (*JUDGMENT_FILES, *(CALIBRATION_FILES if copy_calibration else ()))
                  if (copy_judgments / name).exists()]
    write_pool(output, pool, runs, {
        'case_sources': targets, 'configured_corpora': settings.retrieval.corpora,
        'runs_corpora': {name: (spec if spec == OWN else list(spec)) for name, spec in runs_wanted.items()},
        'adopted_runs': adopted, 'filters': {'legal_domain': 'per case'} if domain_filter else None,
        'pool_depth': pool_depth, 'run_depth': run_depth, 'reranker': settings.rerank.backend,
        'rerank': {'backend': settings.rerank.backend, 'applied': 'inside every retrieval',
                   **{field: getattr(reranker, field, None)
                      for field in ('model', 'device', 'batch_size', 'max_length')}},
        'retrieval': {'k_dense': settings.retrieval.k_dense, 'k_sparse': settings.retrieval.k_sparse,
                      'k_fused': settings.retrieval.k_fused, 'sparse_weight': settings.retrieval.sparse_weight,
                      'rrf_k': settings.retrieval.rrf_k},
        'seconds': round(elapsed, 1), 'judgments_copied_from': str(copy_judgments) if copied else None,
        'copied_files': copied, 'settings_hash': settings.hash()})
    for name in copied:
        shutil.copyfile(copy_judgments / name, output / name)

    key = judge_key(f'ollama:{judge_model}', think=think, prompt_profile=prompt_profile, domain=judge_domain)
    typer.echo(describe_key(key, think=think))
    judged = set()
    if copy_judgments:
        store = JudgmentStore(copy_judgments / 'judgments.jsonl')
        judged = set(store.grades(key)) | set(store.failures(key))
    for name, rows in runs.items():
        missing = len(pending_pairs({name: rows}, judged, pool_depth))
        total = len(pending_pairs({name: rows}, set(), pool_depth))
        typer.echo(f'{name}: {total - missing} of {total} top-{pool_depth} pairs judged, {missing} pending')
    pending = pending_pairs(runs, judged, pool_depth)
    typer.echo(f'{len(pending)} distinct pending pairs across every run '
               f'(~{len(pending) * SECONDS_PER_PAIR / 3600:.1f} h at {SECONDS_PER_PAIR} s/pair)')
    typer.echo(f'wrote {output}: {len(pool)} pairs, {len(runs)} runs')


if __name__ == '__main__':
    app()

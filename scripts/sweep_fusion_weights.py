#!/usr/bin/env python
"""Sweep RRF fusion weights by re-fusing a pool's recorded leg runs; no retrieval, no judging.

The source pool's evaluation applies a single-corpus filter, so runs/dense.jsonl and
runs/sparse.jsonl are the exact ranked lists RRFFuser was given and a differently weighted
hybrid is arithmetic. The new directory holds one hybrid_s<weight> run per variant, the source
pool rows extended with the chunks the variants newly surface, and the copied judgments, so
scripts/judge_relevance.py can grade only the pending pairs and scripts/eval_graded.py can then
compare the variants. The source directory is only read.

Example:
    uv run python scripts/sweep_fusion_weights.py \
      --pool-dir data/eval/relevance/ip-v1-pilot-200-top10 \
      --output data/eval/relevance/ip-v1-pilot-200-sweep
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import typer

from legal_rag.cli import settings_from_options
from legal_rag.evaluation.fusion_sweep import pair_ranks, pending_pairs, refuse
from legal_rag.evaluation.judgment_store import JudgmentStore
from legal_rag.evaluation.pool_extension import extend_pool, gold_document_ids
from legal_rag.evaluation.relevance_judge import JUDGE_THINK, describe_key, judge_key
from legal_rag.evaluation.relevance_pool import JUDGMENT_FILES, read_jsonl, write_pool
from legal_rag.ingestion.corpus import Corpus

app = typer.Typer(add_completion=False)
LEGS = ('dense', 'sparse')
SECONDS_PER_PAIR = 15.3


def _progress(message: str) -> None:
    typer.echo(message, err=True)


def variant_name(sparse_weight: float) -> str:
    """A run file name that reads as a mode column in eval_graded.py, sorting by weight."""
    return f'hybrid_s{sparse_weight:.2f}'


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _needed_chunks(settings, chunk_ids: set[str]) -> dict:
    """Stream the processed corpus once, keeping only the chunks the new pool rows need."""
    found = {}
    for chunk in Corpus.load(settings.processed_corpus_dir()).chunks():
        if chunk.id in chunk_ids:
            found[chunk.id] = chunk
            if len(found) == len(chunk_ids):
                break
    missing = sorted(chunk_ids - set(found))
    if missing:
        raise ValueError(f'{len(missing)} newly surfaced chunks are not in the processed corpus '
                         f'(e.g. {missing[0]}); the pool and the corpus disagree')
    return found


@app.command()
def main(
    pool_dir: Path = typer.Option(..., exists=True, file_okay=False, help='Existing pool directory; read only.'),
    output: Path = typer.Option(..., help='New directory; must not exist.'),
    sparse_weight: list[float] = typer.Option([0.0, 0.25, 0.5, 0.75, 1.0], '--sparse-weight',
                                              help='Repeatable; one hybrid run per weight.'),
    dense_weight: float = typer.Option(1.0, help='Weight of the dense leg in every variant.'),
    rrf_k: int = typer.Option(None, min=1, help='Defaults to retrieval.rrf_k from the settings.'),
    depth: int = typer.Option(10, min=1, help='Pool depth of the new directory.'),
    model: str = typer.Option('qwen3:8b', help='Judge whose existing judgments count as done.'),
    think: bool = typer.Option(JUDGE_THINK, '--think/--no-think'),
    profile: str = typer.Option(None),
    root: Path = typer.Option(None),
    corpus_version: str = typer.Option(None),
) -> None:
    if output.exists():
        raise typer.BadParameter(f'{output} exists; choose a new --output')
    weights = sorted(dict.fromkeys(sparse_weight))
    settings = settings_from_options(profile=profile, root=root, corpus_version=corpus_version)
    rrf_k = settings.retrieval.rrf_k if rrf_k is None else rrf_k
    provenance = json.loads((pool_dir / 'pool.json').read_text(encoding='utf-8'))
    source_depth = int(provenance['pool_depth'])
    if depth > source_depth:
        raise typer.BadParameter(f'depth {depth} > source pool_depth {source_depth}: the source pool holds no '
                                 'rows for deeper ranks and its judgments would not cover them')
    legs = {leg: read_jsonl(pool_dir / 'runs' / f'{leg}.jsonl') for leg in LEGS}
    runs = {**legs, **{variant_name(w): refuse(legs, {'dense': dense_weight, 'sparse': w}, rrf_k=rrf_k, depth=depth)
                       for w in weights}}

    key = judge_key(f'ollama:{model}', think=think)
    _progress(describe_key(key, think=think))
    store = JudgmentStore(pool_dir / 'judgments.jsonl')
    judged = set(store.grades(key)) | set(store.failures(key))
    for name in [*LEGS, *(variant_name(w) for w in weights)]:
        pending = pending_pairs({name: runs[name]}, judged, depth)
        total = len(pair_ranks({name: runs[name]}, depth))
        typer.echo(f'{name}: {total - len(pending)} of {total} top-{depth} pairs judged, {len(pending)} pending')
    pending = pending_pairs(runs, judged, depth)
    typer.echo(f'{len(pending)} distinct pending pairs across every run '
               f'(~{len(pending) * SECONDS_PER_PAIR / 3600:.1f} h at {SECONDS_PER_PAIR} s/pair)')

    source_rows = read_jsonl(pool_dir / 'pool.jsonl')
    ranks = pair_ranks(runs, depth)
    target = Path(provenance['target'])
    target_rows = None
    if target.exists() and _sha256(target) == provenance['target_sha256']:
        target_rows = read_jsonl(target)
    else:
        _progress(f'WARNING: {target} is missing or its sha256 changed; gold flags for newly surfaced chunks '
                  'fall back to the gold rows the source pool happens to hold')
    gold = gold_document_ids(source_rows, target_rows)
    known = {(row['case_id'], row['chunk_id']) for row in source_rows}
    new_chunk_ids = {chunk_id for pair, chunk_id in ((pair, pair[1]) for pair in ranks) if pair not in known}
    _progress(f'{len(new_chunk_ids)} newly surfaced chunks; scanning the processed corpus')
    chunks = _needed_chunks(settings, new_chunk_ids) if new_chunk_ids else {}
    pool = extend_pool(source_rows, ranks, chunks, gold_documents=gold)

    copied = [name for name in JUDGMENT_FILES if (pool_dir / name).exists()]
    write_pool(output, pool, runs, {
        **provenance, 'pool_depth': depth,
        'fusion_sweep': {'rrf_k': rrf_k, 'dense_weight': dense_weight, 'sparse_weights': weights,
                         'legs': list(LEGS), 'variants': {variant_name(w): {'dense': dense_weight, 'sparse': w}
                                                          for w in weights}},
        'derived_from': {'pool_dir': str(pool_dir), 'pool_jsonl_sha256': _sha256(pool_dir / 'pool.jsonl'),
                         'pool_depth': source_depth,
                         'runs_sha256': {leg: _sha256(pool_dir / 'runs' / f'{leg}.jsonl') for leg in LEGS}},
        'judgments_copied_from': str(pool_dir) if copied else None})
    for name in copied:
        shutil.copyfile(pool_dir / name, output / name)
    typer.echo(f'wrote {output}: {len(pool)} pairs, {len(runs)} runs, {len(pending)} pairs still to judge')


if __name__ == '__main__':
    app()

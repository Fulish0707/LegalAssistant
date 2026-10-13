#!/usr/bin/env python
"""Rerank a pool's recorded fused run with a cross-encoder; no retrieval, no judging.

The new directory holds the reranked run beside the baseline runs copied from the source pool,
so the comparison is paired: every run ranks the same candidates, only their order differs. The
pool rows are the source rows extended with the chunks the reranked run newly surfaces, and the
judgments are copied, so scripts/judge_relevance.py grades only the pending pairs and
scripts/eval_graded.py can then compare. The source directory is only read.

Example:
    uv run python scripts/rerank_relevance_pool.py --profile local \
      --pool-dir data/eval/relevance/ip-v1-pilot-200-sweep \
      --output data/eval/relevance/ip-v1-pilot-200-rerank --device cuda
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import typer

from legal_rag.cli import settings_from_options
from legal_rag.evaluation.relevance.fusion_sweep import pair_ranks, pending_pairs
from legal_rag.evaluation.relevance.judgment_store import JudgmentStore
from legal_rag.evaluation.relevance.pool_extension import extend_pool, gold_document_ids, needed_chunks
from legal_rag.evaluation.relevance.relevance_judge import JUDGE_THINK, describe_key, judge_key
from legal_rag.evaluation.relevance.relevance_pool import JUDGMENT_FILES, read_jsonl, write_pool
from legal_rag.evaluation.relevance.rerank_run import rerank_run
from legal_rag.reranking.cross_encoder import CrossEncoderReranker

app = typer.Typer(add_completion=False)
CALIBRATION_FILES = ('calibration.csv', 'calibration.json')
SECONDS_PER_PAIR = 15.3


def _progress(message: str) -> None:
    typer.echo(message, err=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _baseline_map(entries: list[str]) -> dict[str, str]:
    """``name``, or ``output=source`` to copy a source run under a different name.

    Renaming keeps a comparison paired when two directories record the same run name for
    different settings, e.g. ``reranked_512=reranked`` beside a fresh ``reranked_1024``.
    """
    mapping: dict[str, str] = {}
    for entry in entries:
        output_name, _, source_name = entry.partition('=')
        if not output_name or (_ and not source_name):
            raise typer.BadParameter(f'--baseline {entry!r}: expected NAME or OUTPUT=SOURCE')
        mapping[output_name] = source_name or output_name
    return mapping


@app.command()
def main(
    pool_dir: Path = typer.Option(..., exists=True, file_okay=False, help='Existing pool directory; read only.'),
    output: Path = typer.Option(..., help='New directory; must not exist.'),
    fused_run: str = typer.Option('hybrid_s0.75', help='Run whose candidates the reranker sees.'),
    baseline: list[str] = typer.Option(['dense', 'sparse', 'hybrid_s0.75'], '--baseline',
                                       help='Repeatable; runs copied from the source pool for comparison. '
                                            'NAME, or OUTPUT=SOURCE to copy a source run under another name.'),
    run_name: str = typer.Option('reranked', help='Name of the recorded reranked run.'),
    depth: int = typer.Option(10, min=1, help='Pool depth of the new directory.'),
    run_depth: int = typer.Option(100, min=1, help='Candidates reranked per case, and the recorded depth.'),
    rerank_model: str = typer.Option(None, help='Defaults to rerank.model from the settings.'),
    device: str = typer.Option(None, help='Defaults to rerank.device from the settings.'),
    batch_size: int = typer.Option(16, min=1),
    max_length: int = typer.Option(512, min=16),
    judge_model: str = typer.Option('qwen3:8b', help='Judge whose existing judgments count as done.'),
    think: bool = typer.Option(JUDGE_THINK, '--think/--no-think'),
    prompt_profile: str = typer.Option('corpus', help='Relevance instructions: corpus, shared or corpus-experimental.'),
    judge_domain: str = typer.Option(None, help='Use the same domain override as the judging command.'),
    profile: str = typer.Option(None),
    root: Path = typer.Option(None),
    corpus_version: str = typer.Option(None),
) -> None:
    if output.exists():
        raise typer.BadParameter(f'{output} exists; choose a new --output')
    settings = settings_from_options(profile=profile, root=root, corpus_version=corpus_version)
    rerank_model = rerank_model or settings.rerank.model
    device = device or settings.rerank.device
    provenance = json.loads((pool_dir / 'pool.json').read_text(encoding='utf-8'))
    source_depth = int(provenance['pool_depth'])
    if depth > source_depth:
        raise typer.BadParameter(f'depth {depth} > source pool_depth {source_depth}: the source pool holds no '
                                 'rows for deeper ranks and its judgments would not cover them')
    baselines = _baseline_map(baseline)
    if run_name in baselines:
        raise typer.BadParameter(f'--run-name {run_name} collides with a baseline; the reranked run would hide it')
    source_runs = {name: read_jsonl(pool_dir / 'runs' / f'{name}.jsonl')
                   for name in dict.fromkeys([fused_run, *baselines.values()])}
    fused_rows = source_runs[fused_run]

    source_rows = read_jsonl(pool_dir / 'pool.jsonl')
    questions = {row['case_id']: row['question'] for row in source_rows}
    ranked_ids = {hit['chunk_id'] for row in fused_rows for hit in row['ranked'][:run_depth]}
    _progress(f'{len(ranked_ids)} distinct candidate chunks; scanning the processed corpus')
    chunks = needed_chunks(settings, ranked_ids)

    reranker = CrossEncoderReranker(model=rerank_model, device=device, batch_size=batch_size,
                                    max_length=max_length)
    _progress(f'reranking {len(fused_rows)} cases x {run_depth} candidates with {rerank_model} on {device}')
    started = time.perf_counter()
    runs = {**{output_name: source_runs[source_name] for output_name, source_name in baselines.items()},
            run_name: rerank_run(fused_rows, chunks, reranker, questions, depth=run_depth, progress=_progress)}
    elapsed = time.perf_counter() - started
    _progress(f'reranked in {elapsed / 60:.1f} min ({elapsed / max(len(fused_rows), 1):.2f} s/case)')

    key = judge_key(f'ollama:{judge_model}', think=think, prompt_profile=prompt_profile, domain=judge_domain)
    _progress(describe_key(key, think=think))
    store = JudgmentStore(pool_dir / 'judgments.jsonl')
    judged = set(store.grades(key)) | set(store.failures(key))
    for name, rows in runs.items():
        total = len(pair_ranks({name: rows}, depth))
        missing = len(pending_pairs({name: rows}, judged, depth))
        typer.echo(f'{name}: {total - missing} of {total} top-{depth} pairs judged, {missing} pending')
    pending = pending_pairs(runs, judged, depth)
    seconds = len(pending) * SECONDS_PER_PAIR
    cost = f'{seconds / 3600:.1f} h' if seconds >= 3600 else f'{seconds / 60:.1f} min'
    typer.echo(f'{len(pending)} distinct pending pairs across every run (~{cost} at {SECONDS_PER_PAIR} s/pair)')

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
    new_pairs = [pair for pair in ranks if pair not in known]
    _progress(f'{len(new_pairs)} pairs newly enter a top-{depth}')
    pool = extend_pool(source_rows, ranks, {pair[1]: chunks[pair[1]] for pair in new_pairs}, gold_documents=gold)

    copied = [name for name in (*JUDGMENT_FILES, *CALIBRATION_FILES) if (pool_dir / name).exists()]
    write_pool(output, pool, runs, {
        **provenance, 'pool_depth': depth, 'run_depth': run_depth,
        'reranker': 'cross_encoder',
        # settings_hash stays the source pool's: it names the settings that retrieved the runs.
        'rerank': {'model': rerank_model, 'device': device, 'batch_size': batch_size, 'max_length': max_length,
                   'fused_run': fused_run, 'run_name': run_name, 'baselines': baselines,
                   'seconds': round(elapsed, 1), 'settings_hash': settings.hash()},
        'derived_from': {'pool_dir': str(pool_dir), 'pool_jsonl_sha256': _sha256(pool_dir / 'pool.jsonl'),
                         'pool_depth': source_depth,
                         'runs_sha256': {name: _sha256(pool_dir / 'runs' / f'{name}.jsonl')
                                         for name in source_runs}},
        'judgments_copied_from': str(pool_dir) if copied else None, 'copied_files': copied})
    for name in copied:
        shutil.copyfile(pool_dir / name, output / name)
    typer.echo(f'wrote {output}: {len(pool)} pairs, {len(runs)} runs, {len(pending)} pairs still to judge')


if __name__ == '__main__':
    app()

#!/usr/bin/env python
"""Retrieve evidence for a sample of the SFT test split and write finished prompts per condition.

Stage 1 of the adapter evaluation (docs/superpowers/specs/2026-09-26-sft-adapter-eval-design.md).
Each case gets three prompts in the training format: no passages, retrieved passages fitted into
--budget tokens, and the test row's own oracle prompt. Stage 2 (generate_sft_answers.py) only
feeds these messages to the model, so the prompt format is fixed here.

Example:
    uv run python scripts/build_sft_eval_contexts.py --profile local --root "$PWD" \
      --output data/eval/sft/full-v2-200
"""
from __future__ import annotations

import hashlib
import json
import statistics
import time
from pathlib import Path

import typer

from legal_rag.cli import settings_from_options
from legal_rag.training.adapter_eval import CONDITIONS, case_contexts, sample_cases
from legal_rag.factory import build_embedder, build_hybrid_retriever, build_reranker, close_legs
from legal_rag.schemas import RetrievalFilters
from legal_rag.training.dataset import file_hash

app = typer.Typer(add_completion=False)
KEEP = ('id', 'domain', 'document_id', 'question', 'answer', 'messages')


def _test_rows(path: Path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                yield {key: row[key] for key in KEEP}


def _hit(rank: int, scored) -> dict:
    chunk = scored.chunk
    return {'rank': rank, 'chunk_id': chunk.id, 'document_id': chunk.document_id, 'title': chunk.title,
            'order': chunk.order, 'text': chunk.text}


@app.command()
def main(
    output: Path = typer.Option(..., help='New directory for contexts.jsonl and contexts.json.'),
    data_dir: Path = typer.Option(Path('data/training/legal-qa-full-v2'), exists=True, file_okay=False),
    tokenizer_dir: Path = typer.Option(Path('data/training/qwen3-4b-tokenizer'), exists=True, file_okay=False),
    per_domain: int = typer.Option(50, min=1),
    seed: int = typer.Option(42),
    budget: int = typer.Option(768, min=128, help='Prompt tokens for the retrieved condition: 1024 minus an answer reserve.'),
    window_chars: int = typer.Option(480, min=32, help='Passage window, as in the training preparation.'),
    profile: str = typer.Option(None),
    root: Path = typer.Option(None),
) -> None:
    if output.exists():
        raise typer.BadParameter(f'{output} exists; choose a new --output')
    manifest = json.loads((data_dir / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete':
        raise typer.BadParameter(f'{data_dir} is not a complete preparation')
    test_path = data_dir / 'test.jsonl'
    if file_hash(test_path) != manifest['files']['test.jsonl']:
        raise typer.BadParameter(f'{test_path} changed since preparation')
    for name, expected in manifest['tokenizer_files'].items():
        if file_hash(tokenizer_dir / name) != expected:
            raise typer.BadParameter(f'tokenizer file {name} differs from the one the data was prepared with')

    from transformers import AutoTokenizer  # heavy; only after the cheap checks

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    cases = sample_cases(_test_rows(test_path), per_domain=per_domain, seed=seed)
    settings = settings_from_options(profile=profile, root=root)
    k = settings.retrieval.k_final
    typer.echo(f'{len(cases)} cases; corpora {settings.retrieval.corpora}, k_final {k}, '
               f'sparse_weight {settings.retrieval.sparse_weight}, rerank {settings.rerank.backend}', err=True)

    retriever = build_hybrid_retriever(settings, embedder=build_embedder(settings), reranker=build_reranker(settings))
    output.mkdir(parents=True)
    started = time.perf_counter()
    written = []
    try:
        with (output / 'contexts.jsonl').open('w', encoding='utf-8') as handle:
            for number, row in enumerate(cases, 1):
                hits = retriever.retrieve(row['question'], k, RetrievalFilters(legal_domain=row['domain']))
                context = case_contexts(row, [_hit(rank, hit) for rank, hit in enumerate(hits, 1)],
                                        tokenizer=tokenizer, budget=budget, window_chars=window_chars)
                handle.write(json.dumps(context, ensure_ascii=False) + '\n')
                handle.flush()
                written.append(context)
                if number % 20 == 0 or number == len(cases):
                    typer.echo(f'  {number}/{len(cases)} retrieved', err=True)
    finally:
        close_legs(retriever)
    elapsed = time.perf_counter() - started

    tokens = {c: [ctx['conditions'][c]['prompt_tokens'] for ctx in written] for c in CONDITIONS}
    stats = {
        'cases': len(written),
        'gold_retrieved': sum(ctx['gold_retrieved'] for ctx in written),
        'retrieved_without_passages': sum(not ctx['conditions']['retrieved']['passages'] for ctx in written),
        'retrieved_passages_mean': statistics.mean(len(ctx['conditions']['retrieved']['passages']) for ctx in written),
        'prompt_tokens': {c: {'mean': statistics.mean(v), 'max': max(v)} for c, v in tokens.items()},
    }
    provenance = {
        'data_dir': str(data_dir.resolve()), 'manifest_sha256': file_hash(data_dir / 'manifest.json'),
        'test_sha256': manifest['files']['test.jsonl'], 'tokenizer_dir': str(tokenizer_dir.resolve()),
        'tokenizer_revision': manifest['tokenizer_revision'], 'max_length': manifest['max_length'],
        'per_domain': per_domain, 'seed': seed, 'budget': budget, 'window_chars': window_chars,
        'retrieval': {'corpora': settings.retrieval.corpora, 'k_final': k, 'k_dense': settings.retrieval.k_dense,
                      'k_sparse': settings.retrieval.k_sparse, 'k_fused': settings.retrieval.k_fused,
                      'sparse_weight': settings.retrieval.sparse_weight, 'rrf_k': settings.retrieval.rrf_k,
                      'rerank': settings.rerank.backend, 'filter': 'legal_domain per case', 'query': 'raw question'},
        'settings_hash': settings.hash(), 'seconds': round(elapsed, 1), 'stats': stats,
        'contexts_sha256': hashlib.sha256((output / 'contexts.jsonl').read_bytes()).hexdigest(),
    }
    (output / 'contexts.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    typer.echo(json.dumps(stats, ensure_ascii=False, indent=2))
    typer.echo(f'wrote {output} in {elapsed / 60:.1f} min')


if __name__ == '__main__':
    app()

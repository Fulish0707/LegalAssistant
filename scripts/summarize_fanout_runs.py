#!/usr/bin/env python
"""Compare a pool directory's runs without a judge: gold documents found, and which corpus answered.

Both summaries come from the recorded runs and the targets alone, so they can be read before any
pair is judged — unlike the graded metrics, which count every unjudged chunk as irrelevant and so
penalise exactly the run that surfaced new chunks.

Example:
    uv run python scripts/summarize_fanout_runs.py --pool-dir data/eval/relevance/ip-200-fanout \
      --profile local --corpus ip-v1 --corpus criminal-v1 --corpus civil-v1 --corpus administrative-v1
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

from legal_rag.cli import settings_from_options
from legal_rag.evaluation.diagnosis.crossdomain_target import corpus_document_ids
from legal_rag.evaluation.relevance.fanout_summary import corpus_mix, gold_document_metrics
from legal_rag.evaluation.relevance.relevance_pool import read_jsonl
from legal_rag.factory import corpus_settings

app = typer.Typer(add_completion=False)


def _table(title: str, summary: dict[str, dict]) -> str:
    runs = list(summary)
    rows = list(summary[runs[0]])
    lines = [f'## {title}', '', '| Metric | ' + ' | '.join(runs) + ' |', '|---|' + '---:|' * len(runs)]
    for row in rows:
        values = [summary[run][row] for run in runs]
        lines.append(f'| {row} | ' + ' | '.join(
            str(value) if isinstance(value, int) else f'{value:.3f}' for value in values) + ' |')
    return '\n'.join(lines)


@app.command()
def main(
    pool_dir: Path = typer.Option(..., exists=True, file_okay=False),
    corpus: list[str] = typer.Option([], '--corpus', help='Repeatable; corpora to attribute hits to.'),
    depth: int = typer.Option(10, min=1),
    profile: str = typer.Option(None),
    root: Path = typer.Option(None),
) -> None:
    provenance = json.loads((pool_dir / 'pool.json').read_text(encoding='utf-8'))
    gold: dict[str, set[str]] = {}
    for source in provenance['case_sources'].values():
        for row in read_jsonl(Path(source['target'])):
            gold[row['id']] = set(row['relevant_document_ids'])
    runs = {path.stem: read_jsonl(path) for path in sorted((pool_dir / 'runs').glob('*.jsonl'))}
    typer.echo(f'# {pool_dir.name}: runs compared without a judge\n')
    typer.echo(_table(f'Gold document at depth {depth}', gold_document_metrics(runs, gold, depth=depth)))
    if corpus:
        settings = settings_from_options(profile=profile, root=root)
        corpora = {version: corpus_document_ids(corpus_settings(settings, version).processed_corpus_dir())
                   for version in corpus}
        typer.echo('\n' + _table(f'Share of top-{depth} slots by corpus', corpus_mix(runs, corpora, depth=depth)))


if __name__ == '__main__':
    app()

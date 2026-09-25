#!/usr/bin/env python
"""Write a blind adapter-vs-base pairwise.csv for one condition of an SFT evaluation.

Stage 3 of the adapter evaluation. The CSV shows the question, the reference answer and the two
answers as A and B; pairwise_key.json holds which side is the adapter. Fill the winner column
(A / B / tie), then score with scripts/score_answers.py --side adapter.

Example:
    uv run python scripts/compare_sft_answers.py --contexts-dir data/eval/sft/full-v2-200 \
      --condition retrieved --output data/eval/sft/full-v2-200/pairwise-retrieved
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import typer

from legal_rag.evaluation.answer_compare import CSV_FIELDS_WITH_REFERENCE, pairwise_rows
from legal_rag.evaluation.relevance_pool import read_jsonl
from legal_rag.training.adapter_eval import CONDITIONS, pairwise_answers

app = typer.Typer(add_completion=False)


@app.command()
def main(
    contexts_dir: Path = typer.Option(..., exists=True, file_okay=False),
    output: Path = typer.Option(..., help='New directory for pairwise.csv and pairwise_key.json.'),
    answers: Path = typer.Option(None, help='Defaults to <contexts-dir>/answers.jsonl.'),
    condition: str = typer.Option('retrieved', help=f'One of {", ".join(CONDITIONS)}.'),
    seed: int = typer.Option(42),
) -> None:
    if condition not in CONDITIONS:
        raise typer.BadParameter(f'--condition must be one of {CONDITIONS}')
    if output.exists():
        raise typer.BadParameter(f'{output} exists; choose a new --output')
    answers = answers or contexts_dir / 'answers.jsonl'
    contexts = read_jsonl(contexts_dir / 'contexts.jsonl')
    generation = json.loads(answers.with_suffix('.json').read_text(encoding='utf-8'))
    try:
        rows, key = pairwise_rows(pairwise_answers(contexts, read_jsonl(answers), condition=condition), seed=seed)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    output.mkdir(parents=True)
    with (output / 'pairwise.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS_WITH_REFERENCE)
        writer.writeheader()
        writer.writerows(rows)
    meta = json.loads((contexts_dir / 'contexts.json').read_text(encoding='utf-8'))
    (output / 'pairwise_key.json').write_text(json.dumps(
        {'key': key, 'sides': {'base': {'model': generation['model_id']}, 'adapter': {'adapter': generation['adapter']}},
         'model': f"{generation['model_id']} (+ adapter)", 'k': meta['retrieval']['k_final'], 'condition': condition,
         'seed': seed, 'contexts_sha256': meta['contexts_sha256'], 'adapter_sha256': generation['adapter_sha256']},
        ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    typer.echo(f'wrote {output}: {len(rows)} pairs under {condition}')
    typer.echo('fill the winner column (A / B / tie), then run scripts/score_answers.py '
               f'--comparison-dir {output} --side adapter')


if __name__ == '__main__':
    app()

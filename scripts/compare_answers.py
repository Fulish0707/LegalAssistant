#!/usr/bin/env python
"""Generate one answer per corpus for the same questions, for a blind pairwise comparison.

Evidence comes from each pool's recorded ranking, so nothing is retrieved again and both sides
answer from exactly the runs that were measured. The CSV hides which corpus wrote which answer;
`pairwise_key.json` holds the mapping for scoring afterwards.

Example:
    uv run python scripts/compare_answers.py \
      --side ip-v1=data/eval/relevance/ip-v1-pilot-200-rerank-k50:reranked_k50 \
      --side ip-v2=data/eval/relevance/ip-v2-rerank:hybrid \
      --output data/eval/answers/v1-vs-v2 --limit 60 --model qwen3:8b
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import typer

from legal_rag.evaluation.answer_compare import (
    CSV_FIELDS, evidence_bundle, pairwise_rows, render_answer, sample_cases)
from legal_rag.evaluation.relevance_pool import read_jsonl
from legal_rag.llm import OllamaChatModel
from legal_rag.rag.generator import GenerationError, Generator

app = typer.Typer(add_completion=False)


def _side(entry: str) -> tuple[str, Path, str]:
    name, _, rest = entry.partition('=')
    directory, _, run = rest.partition(':')
    if not (name and directory and run):
        raise typer.BadParameter(f'--side {entry!r}: expected NAME=POOL_DIR:RUN')
    return name, Path(directory), run


@app.command()
def main(
    side: list[str] = typer.Option(..., '--side', help='NAME=POOL_DIR:RUN, exactly twice.'),
    output: Path = typer.Option(..., help='New directory; an existing one is refused.'),
    limit: int = typer.Option(60, min=1, help='Questions to answer, sampled from the shared cases.'),
    k: int = typer.Option(5, min=1, help='Chunks given to the generator, as retrieval.k_final does.'),
    model: str = typer.Option('qwen3:8b'),
    base_url: str = typer.Option('http://localhost:11434'),
    num_ctx: int = typer.Option(16384, min=2048, help='Context window; ip-v1 evidence runs to ~24k characters.'),
    max_tokens: int = typer.Option(
        2048, min=128, help='A 1024 limit truncated 18% of answers mid-JSON in the first run.'),
    resume: bool = typer.Option(False, help='Keep answers already in the output and only fill the gaps.'),
    seed: int = typer.Option(42),
) -> None:
    if len(side) != 2:
        raise typer.BadParameter('exactly two --side entries are required')
    answered_path = output / 'answers.jsonl'
    if output.exists() and not resume:
        raise typer.BadParameter(f'{output} exists; choose a new --output or pass --resume')
    done = {row['case_id']: row for row in read_jsonl(answered_path)} if resume and answered_path.exists() else {}
    sides = [_side(entry) for entry in side]

    pools, runs = {}, {}
    for name, directory, run in sides:
        pools[name] = {(row['case_id'], row['chunk_id']): row for row in read_jsonl(directory / 'pool.jsonl')}
        runs[name] = {row['case_id']: row['ranked'] for row in read_jsonl(directory / 'runs' / f'{run}.jsonl')}
    shared = set.intersection(*(set(run) for run in runs.values()))
    cases = sample_cases(shared, limit=limit, seed=seed)
    questions = {row['case_id']: row['question'] for name in pools for row in pools[name].values()}
    typer.echo(f'{len(shared)} shared cases; answering {len(cases)} with {model}', err=True)

    chat = OllamaChatModel(model=model, base_url=base_url, timeout=600.0, think=False, num_ctx=num_ctx)
    generator = Generator(chat, max_tokens=max_tokens)
    answers, failures = list(done.values()), []
    output.mkdir(parents=True, exist_ok=True)
    # Answers are appended as they are generated: a run of this length should never be lost to a
    # crash, and --resume then starts from what is already on disk.
    handle = answered_path.open('a', encoding='utf-8')
    if done:
        typer.echo(f'resuming: {len(done)} answers already on disk', err=True)
    for number, case_id in enumerate(cases, 1):
        if case_id in done:
            continue
        question = questions[case_id]
        row = {'case_id': case_id, 'question': question}
        for name in pools:
            bundle = evidence_bundle(pools[name], case_id, runs[name][case_id], k)
            try:
                row[name] = render_answer(generator.generate(question, bundle))
            except GenerationError as exc:
                failures.append({'case_id': case_id, 'side': name, 'error': str(exc)[:200]})
                row = None
                break
        if row is not None:
            answers.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            handle.flush()
        if number % 5 == 0 or number == len(cases):
            typer.echo(f'  {number}/{len(cases)} answered, {len(failures)} failed', err=True)

    handle.close()
    rows, key = pairwise_rows(answers, seed=seed)
    # Sides are seeded per case, so verdicts already collected still describe the same pair.
    verdicts = {}
    csv_path = output / 'pairwise.csv'
    if csv_path.exists():
        with csv_path.open(newline='', encoding='utf-8-sig') as existing:
            verdicts = {row['case_id']: row for row in csv.DictReader(existing) if (row.get('winner') or '').strip()}
    for row in rows:
        kept = verdicts.get(row['case_id'])
        if kept:
            row['winner'], row['reason'] = kept['winner'], kept.get('reason', '')
    if verdicts:
        typer.echo(f'carried {len(verdicts)} verdicts forward', err=True)
    with csv_path.open('w', newline='', encoding='utf-8-sig') as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (output / 'pairwise_key.json').write_text(json.dumps(
        {'key': key, 'sides': {name: {'pool': str(directory), 'run': run} for name, directory, run in sides},
         'model': model, 'k': k, 'num_ctx': num_ctx, 'seed': seed, 'failures': failures},
        ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    typer.echo(f'wrote {output}: {len(rows)} pairs, {len(failures)} generation failures')
    typer.echo('fill the winner column (A / B / tie) in pairwise.csv, then run scripts/score_answers.py')


if __name__ == '__main__':
    app()

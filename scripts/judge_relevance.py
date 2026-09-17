#!/usr/bin/env python
"""Grade pool pairs with an Ollama judge, appending to judgments.jsonl. Safe to interrupt and rerun.

The human labels calibration.csv first; the judge refuses to run until every calibration pair
has a human grade (override with --skip-calibration-check).

Example:
    uv run python scripts/judge_relevance.py --pool-dir data/eval/relevance/ip-v1-pilot-200 --calibration-only
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

from legal_rag.evaluation.calibration import require_human_grades
from legal_rag.evaluation.judgment_store import JudgeKey, JudgmentStore
from legal_rag.evaluation.relevance_judge import JUDGE_NUM_CTX, PROMPT_SHA256, judge_pool
from legal_rag.evaluation.relevance_pool import read_jsonl

app = typer.Typer(add_completion=False)


@app.command()
def main(
    pool_dir: Path = typer.Option(..., exists=True, file_okay=False),
    model: str = typer.Option('qwen3:8b'),
    base_url: str = typer.Option('http://localhost:11434'),
    calibration_only: bool = typer.Option(False, help='Judge only the pairs in calibration.json.'),
    pool_file: Path = typer.Option(None, exists=True, dir_okay=False,
                                   help='Another pool file, e.g. from a new run; defaults to pool.jsonl.'),
    skip_calibration_check: bool = typer.Option(
        False, help='Judge even though the human calibration grades are incomplete.'),
) -> None:
    try:
        if not skip_calibration_check:
            require_human_grades(pool_dir)
        pairs = read_jsonl(pool_file or pool_dir / 'pool.jsonl')
        if calibration_only:
            manifest = json.loads((pool_dir / 'calibration.json').read_text(encoding='utf-8'))
            wanted = {pair['pair_id'] for pair in manifest['pairs']}
            pairs = [p for p in pairs if f"{p['case_id']}|{p['chunk_id']}" in wanted]
    except (ValueError, OSError) as exc:
        typer.echo(f'refusing to judge: {exc}', err=True)
        raise typer.Exit(1) from exc

    # Imported late so a refusal never touches the model client.
    from legal_rag.llm import OllamaChatModel

    key = JudgeKey(f'ollama:{model}', PROMPT_SHA256)
    typer.echo(f'judge {key.model} prompt {key.prompt_sha256[:12]} num_ctx {JUDGE_NUM_CTX}: {len(pairs)} pairs',
               err=True)
    chat = OllamaChatModel(model=model, base_url=base_url, timeout=300.0, think=False, num_ctx=JUDGE_NUM_CTX)
    counts = judge_pool(pairs, chat, JudgmentStore(pool_dir / 'judgments.jsonl'),
                        progress=lambda message: typer.echo(message, err=True))
    typer.echo(json.dumps(counts))


if __name__ == '__main__':
    app()

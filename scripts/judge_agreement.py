#!/usr/bin/env python
"""Compare human calibration grades with the judge's grades.

Writes agreement.json and calibration_review.csv (pairs where human and judge differ) into the
pool directory. Refuses unless every calibration pair has a human grade, and exits 1 when any
human-graded pair has no judge grade or binary grade-2 kappa is below the threshold.

Example:
    uv run python scripts/judge_agreement.py --pool-dir data/eval/relevance/ip-v1-pilot-200
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer

from legal_rag.evaluation.calibration import (
    PASS_KAPPA, agreement, disagreement_rows, judge_passes, missing_llm_message, read_human_grades,
    require_human_grades, write_review)
from legal_rag.evaluation.judgment_store import JudgeKey, JudgmentStore
from legal_rag.evaluation.relevance_judge import PROMPT_SHA256
from legal_rag.evaluation.relevance_pool import read_jsonl

app = typer.Typer(add_completion=False)


@app.command()
def main(pool_dir: Path = typer.Option(..., exists=True, file_okay=False),
         model: str = typer.Option('qwen3:8b')) -> None:
    key = JudgeKey(f'ollama:{model}', PROMPT_SHA256)
    csv_path = pool_dir / 'calibration.csv'
    try:
        require_human_grades(pool_dir)
        human = read_human_grades(csv_path)
        store = JudgmentStore(pool_dir / 'judgments.jsonl')
        gold = {(r['case_id'], r['chunk_id']) for r in read_jsonl(pool_dir / 'pool.jsonl') if r['is_gold_chunk']}
        result = agreement(human, store.grades(key), gold_pairs=gold)
        review = disagreement_rows(csv_path, store.rows(key))
    except (ValueError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    report = {'judge_model': key.model, 'prompt_sha256': key.prompt_sha256,
              'computed_at': datetime.now(timezone.utc).isoformat(), 'pass_kappa': PASS_KAPPA, **result}
    (pool_dir / 'agreement.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                                             encoding='utf-8')
    write_review(pool_dir / 'calibration_review.csv', review)
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    typer.echo(f'wrote agreement.json and calibration_review.csv ({len(review)} disagreements)', err=True)
    if not judge_passes(result):
        if result['missing_llm']:
            typer.echo(f'FAIL: {missing_llm_message(result)}')
        else:
            typer.echo(f'FAIL: binary grade-2 kappa below {PASS_KAPPA}; revise the prompt before the full run')
        raise typer.Exit(1)
    typer.echo('PASS: judge usable')


if __name__ == '__main__':
    app()

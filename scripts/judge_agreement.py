#!/usr/bin/env python
"""Compare reference calibration grades with the judge's grades.

Writes agreement-<hash12>.json and calibration_review-<hash12>.csv (pairs where reference and judge
differ) into the pool directory, one pair of files per judge key. Refuses unless every calibration
pair has a reference grade, and exits 1 when any graded pair has no judge grade or binary kappa at
the pass boundary (--pass-boundary: grade >= 1, relevant vs 0; or grade >= 2, the default) is below
the threshold. Both binary statistics are always reported (binary_relevant, binary_grade2).
Agreement is also reported on the judge's grades before quote verification
(``agreement_before_quote_check``), with ``quote_downgrades`` counting the calibration pairs whose
grade 2 became 1.

Example:
    uv run python scripts/judge_agreement.py --pool-dir data/eval/relevance/ip-v1-pilot-200 \
      --reference-source "GPT 6 Astra" --think --pass-boundary 1
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer

from legal_rag.evaluation.calibration import (
    BOUNDARY_KEYS, PASS_KAPPA, agreement, agreement_paths, calibration_fingerprint, disagreement_rows,
    judge_passes, missing_llm_message, model_grades, quote_downgrades, read_reference_grades, require_reference_grades,
    write_review)
from legal_rag.evaluation.judgment_store import JudgmentStore
from legal_rag.evaluation.relevance_judge import (
    JUDGE_THINK, QUOTE_RULE_VERSION, describe_key, generation_settings, judge_key)
from legal_rag.evaluation.relevance_pool import read_jsonl

app = typer.Typer(add_completion=False)


@app.command()
def main(pool_dir: Path = typer.Option(..., exists=True, file_okay=False),
         model: str = typer.Option('qwen3:8b'),
         think: bool = typer.Option(JUDGE_THINK, '--think/--no-think', help='Check judgments made with thinking on.'),
         reference_source: str = typer.Option(
             'codex', help='Who produced the codex_grade column (e.g. "GPT 6 Astra"); recorded in the report.'),
         pass_boundary: int = typer.Option(
             2, min=1, max=2, help='Pass on the binary statistic at grade >= this: 1 (relevant vs 0) or 2 (grade 2 vs 0/1).'),
         pass_metric: str = typer.Option(
             'kappa', help="Agreement statistic the pass uses: 'kappa', or 'ac1' (Gwet) when the sample's "
                           'prevalence is lopsided. Choose it before reading the numbers.'),
         prompt_profile: str = typer.Option('corpus', help='Relevance instructions: corpus, shared or corpus-experimental.'),
         judge_domain: str = typer.Option(None, help='Use the same domain override as the judging command.'),
         ) -> None:
    key = judge_key(f'ollama:{model}', think=think, prompt_profile=prompt_profile, domain=judge_domain)
    typer.echo(describe_key(key, think=think))
    typer.echo(f'reference labels: {reference_source}')
    csv_path = pool_dir / 'calibration.csv'
    try:
        require_reference_grades(pool_dir)
        reference = read_reference_grades(csv_path)
        store = JudgmentStore(pool_dir / 'judgments.jsonl')
        gold = {(r['case_id'], r['chunk_id']) for r in read_jsonl(pool_dir / 'pool.jsonl') if r['is_gold_chunk']}
        llm_rows = store.rows(key)
        result = agreement(reference, store.grades(key), gold_pairs=gold)
        before = agreement(reference, model_grades(llm_rows), gold_pairs=gold)
        review = disagreement_rows(csv_path, llm_rows)
    except (ValueError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    report = {'judge_model': key.model, 'prompt_sha256': key.prompt_sha256, 'think': think,
              'prompt_profile': prompt_profile, 'question_domain_override': judge_domain,
              'generation': generation_settings(think=think), 'quote_rule_version': QUOTE_RULE_VERSION,
              'reference_source': reference_source,
              'computed_at': datetime.now(timezone.utc).isoformat(), 'pass_kappa': PASS_KAPPA,
              'pass_boundary': pass_boundary, 'pass_metric': pass_metric,
              **calibration_fingerprint(pool_dir), **result,
              'quote_downgrades': quote_downgrades(llm_rows, reference.keys()),
              'agreement_before_quote_check': before}
    report_path, review_path = agreement_paths(pool_dir, key.prompt_sha256)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_review(review_path, review)
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    typer.echo(f'wrote {report_path.name} and {review_path.name} ({len(review)} disagreements)', err=True)
    score = result[BOUNDARY_KEYS[pass_boundary]].get(pass_metric)
    stat = f"binary grade≥{pass_boundary} {pass_metric} {'—' if score is None else f'{score:.3f}'}"
    if not judge_passes(result, pass_boundary, metric=pass_metric):
        if result['missing_llm']:
            typer.echo(f'FAIL: {missing_llm_message(result)}')
        else:
            typer.echo(f'FAIL: {stat} below {PASS_KAPPA}; revise the prompt before the full run')
        raise typer.Exit(1)
    typer.echo(f'PASS: {stat} >= {PASS_KAPPA}; judge usable at this boundary')


if __name__ == '__main__':
    app()

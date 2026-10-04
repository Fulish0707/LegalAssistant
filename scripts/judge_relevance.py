#!/usr/bin/env python
"""Grade pool pairs with an Ollama judge, appending to judgments.jsonl. Safe to interrupt and rerun.

The reference labeller fills codex_grade in calibration.csv first; the judge refuses to run until every
calibration pair has a codex_grade (override with --skip-calibration-check). Runs that pass the check append a
provenance line (calibration CSV hash, model digest, flags, counts) to judge_runs.jsonl.

Example:
    uv run python scripts/judge_relevance.py --pool-dir data/eval/relevance/ip-v1-pilot-200 --calibration-only
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import typer

from legal_rag.evaluation.calibration import calibration_fingerprint, require_reference_grades
from legal_rag.evaluation.judgment_store import JudgmentStore
from legal_rag.evaluation.relevance_judge import (
    JUDGE_THINK, append_judge_run, describe_key, generation_settings, judge_key, judge_pool, judge_run_record,
    ollama_model_digest)
from legal_rag.evaluation.relevance_pool import read_jsonl
from legal_rag.evaluation.relevance_prompts import attach_question_domains, guidance_domain, question_domain, validate_profile

app = typer.Typer(add_completion=False)


@app.command()
def main(
    pool_dir: Path = typer.Option(..., exists=True, file_okay=False),
    model: str = typer.Option('qwen3:8b'),
    base_url: str = typer.Option('http://localhost:11434'),
    calibration_only: bool = typer.Option(False, help='Judge only the pairs in calibration.json.'),
    pool_file: Path = typer.Option(None, exists=True, dir_okay=False,
                                   help='Pairs to judge instead of pool.jsonl; must come from the same target.'),
    retry_failed: bool = typer.Option(False, help='Judge again pairs whose stored judgment failed to parse.'),
    skip_calibration_check: bool = typer.Option(
        False, help='Judge even though the codex_grade calibration labels are incomplete.'),
    think: bool = typer.Option(JUDGE_THINK, '--think/--no-think',
                               help='Let the model think first; a separate judgment key with more output tokens.'),
    prompt_profile: str = typer.Option('corpus', help='Relevance instructions: corpus, shared or corpus-experimental.'),
    judge_domain: str = typer.Option(None, help='Override question domain; criminal, civil, administrative, ip, mixed or unknown.'),
) -> None:
    try:
        validate_profile(prompt_profile, judge_domain)
        if not skip_calibration_check:
            require_reference_grades(pool_dir)
        # Hashed when the guard passes, so later edits to the labels do not change the record.
        calibration_csv_sha256 = calibration_fingerprint(pool_dir)['calibration_csv_sha256']
        pairs = read_jsonl(pool_file or pool_dir / 'pool.jsonl')
        if calibration_only:
            manifest = json.loads((pool_dir / 'calibration.json').read_text(encoding='utf-8'))
            wanted = {pair['pair_id'] for pair in manifest['pairs']}
            pairs = [p for p in pairs if f"{p['case_id']}|{p['chunk_id']}" in wanted]
        if prompt_profile != 'shared' and judge_domain is None:
            pairs = attach_question_domains(pairs, pool_dir)
    except (ValueError, OSError) as exc:
        typer.echo(f'refusing to judge: {exc}', err=True)
        raise typer.Exit(1) from exc

    # Imported late so a refusal never touches the model client.
    from legal_rag.llm import OllamaChatModel

    key = judge_key(f'ollama:{model}', think=think, prompt_profile=prompt_profile, domain=judge_domain)
    routes = Counter((question_domain(p, judge_domain) or 'shared_fallback')
                     if prompt_profile != 'shared' else 'shared_fallback' for p in pairs)
    typer.echo(f'prompt profile {prompt_profile}; question routes {dict(routes)}', err=True)
    guidance_routes = Counter(guidance_domain(p, profile=prompt_profile, domain=judge_domain) or 'shared_fallback'
                              for p in pairs)
    typer.echo(f'applied guidance {dict(guidance_routes)}', err=True)
    settings = generation_settings(think=think)
    typer.echo(f"{describe_key(key, think=think)} max_tokens {settings['max_tokens']} num_ctx {settings['num_ctx']}: "
               f'{len(pairs)} pairs', err=True)
    digest = ollama_model_digest(base_url, model)
    chat = OllamaChatModel(model=model, base_url=base_url, timeout=300.0, think=think, num_ctx=settings['num_ctx'])
    if chat.name != key.model:
        raise RuntimeError(f'model name {chat.name} does not match judge key {key.model}')
    counts = judge_pool(pairs, chat, JudgmentStore(pool_dir / 'judgments.jsonl'), think=think,
                        progress=lambda message: typer.echo(message, err=True), retry_failed=retry_failed,
                        prompt_profile=prompt_profile, domain=judge_domain)
    if not skip_calibration_check:
        record = judge_run_record(
            key, calibration_csv_sha256=calibration_csv_sha256,
            calibration_only=calibration_only, retry_failed=retry_failed, pairs=len(pairs), counts=counts,
            model_digest=digest, think=think, prompt_profile=prompt_profile, domain=judge_domain)
        record['question_domains'] = dict(routes)
        record['guidance_domains'] = dict(guidance_routes)
        append_judge_run(pool_dir, record)
    typer.echo(json.dumps(counts))


if __name__ == '__main__':
    app()

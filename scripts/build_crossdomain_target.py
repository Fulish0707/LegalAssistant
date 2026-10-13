#!/usr/bin/env python
"""Draw a reproducible evaluation target from domains other than intellectual property.

Case ids are the ones load_diagnostic_rows assigns, so each per-domain targets/<domain>.jsonl
reproduces its cases exactly; the script checks that before writing. The combined target.jsonl
is the union for reference and cannot be loaded for a single domain (its other-domain pairs
would be reported as unmatched), which is why the per-domain files exist.

Example:
    uv run python scripts/build_crossdomain_target.py --profile local \
      --domain criminal=criminal-v1 --domain civil=civil-v1 --domain administrative=administrative-v1 \
      --qa-root /home/layke/Documents/legal-rag/data/processed --per-domain 20 --seed 42 \
      --output data/eval/targets/crossdomain-60
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import typer

from legal_rag.cli import settings_from_options
from legal_rag.evaluation.diagnosis.crossdomain_target import (
    corpus_document_ids, drop_ambiguous_pairs, sample_cases, sidecar_rows, target_rows)
from legal_rag.evaluation.diagnosis.diagnostic_runner import load_diagnostic_rows
from legal_rag.factory import corpus_settings

app = typer.Typer(add_completion=False)


def _progress(message: str) -> None:
    typer.echo(message, err=True)


def _corpora(entries: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for entry in entries:
        domain, sep, version = entry.partition('=')
        if not (domain and sep and version):
            raise typer.BadParameter(f'--domain {entry!r}: expected DOMAIN=CORPUS_VERSION')
        mapping[domain] = version
    return mapping


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


@app.command()
def main(
    domain: list[str] = typer.Option(..., '--domain', help='Repeatable DOMAIN=CORPUS_VERSION.'),
    output: Path = typer.Option(..., help='New directory; existing directories are never overwritten.'),
    qa_root: Path = typer.Option(..., exists=True, file_okay=False,
                                 help='Directory holding <domain>/qa.jsonl.'),
    per_domain: int = typer.Option(20, min=1),
    seed: int = typer.Option(42),
    profile: str = typer.Option(None),
    root: Path = typer.Option(None),
) -> None:
    if output.exists():
        raise typer.BadParameter(f'{output} exists; choose a new --output')
    corpora = _corpora(domain)
    settings = settings_from_options(profile=profile, root=root)
    cases, provenance = [], {}
    for name, version in corpora.items():
        qa_file = qa_root / name / 'qa.jsonl'
        rows, _, source = load_diagnostic_rows(qa_file, domain=name)
        unique = drop_ambiguous_pairs(rows)
        _progress(f'{name}: {len(rows)} QA rows, {len(unique)} with an unambiguous question/document pair; '
                  f'reading document ids from {version}')
        with_chunks = corpus_document_ids(corpus_settings(settings, version).processed_corpus_dir())
        drawn = sample_cases(unique, limit=per_domain, seed=seed, has_chunks=with_chunks.__contains__)
        _write(output / 'targets' / f'{name}.jsonl', target_rows(drawn))
        reloaded, _, _ = load_diagnostic_rows(qa_file, domain=name, dataset=output / 'targets' / f'{name}.jsonl')
        if [row['case_id'] for row in reloaded] != [case['case_id'] for case in drawn]:
            raise typer.BadParameter(f'{name}: the written target does not reproduce the drawn case ids')
        cases += drawn
        provenance[name] = {'corpus_version': version, 'qa_file': source['qa_file'],
                            'qa_rows': len(rows), 'unambiguous_rows': len(unique),
                            'documents_with_chunks': len(with_chunks), 'cases': len(drawn)}
        _progress(f'{name}: drew {len(drawn)} cases, reproduced from targets/{name}.jsonl')
    cases.sort(key=lambda case: case['case_id'])
    _write(output / 'target.jsonl', target_rows(cases))
    _write(output / 'target_cases.jsonl', sidecar_rows(cases, corpora))
    (output / 'target.json').write_text(json.dumps(
        {'seed': seed, 'per_domain': per_domain, 'cases': len(cases), 'domains': provenance,
         'target_sha256': hashlib.sha256((output / 'target.jsonl').read_bytes()).hexdigest()},
        ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    typer.echo(f'wrote {output}: {len(cases)} cases across {len(corpora)} domains')


if __name__ == '__main__':
    app()

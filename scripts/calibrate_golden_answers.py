#!/usr/bin/env python
"""Regrade frozen original/structural answers with v3 and report paired outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import shutil


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'data/eval/reports/chat_pipeline/structural-golden-80-2026-10-03'


def prepare_output(source, output):
    """Copy frozen inputs and historical grades; never mix judge configurations."""
    if source == output or source in output.parents:
        raise ValueError('Output must be a separate sibling directory, outside the source run.')
    files = {}
    for arm in ('original', 'structural'):
        names = ['answer_pairs.jsonl', 'verify.jsonl', 'contexts.jsonl',
                 'answer_judge-v2.json', 'answer_judgments-v2.jsonl']
        names += [p.name for p in (source / arm).glob('answer_judge_retry-*.json')
                  if not json.loads(p.read_text()).get('policy', '').startswith('answer-checklist')]
        for name in names:
            path = source / arm / name
            if path.exists():
                files[f'{arm}/{name}'] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = {'source_run': str(source), 'files_sha256': files}
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / 'frozen-inputs.json'
    if manifest.exists() and json.loads(manifest.read_text()) != provenance:
        raise ValueError('Frozen source inputs changed; select a new --out directory.')
    for relative, digest in files.items():
        dest = output / relative
        if dest.exists():
            if hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
                raise ValueError(f'Copied frozen input changed: {relative}')
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, dest)
    manifest.write_text(json.dumps(provenance, indent=2)+'\n')


def report(run):
    from legal_rag.evaluation.answer_reporting import write_calibration_report
    write_calibration_report(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('judge', 'report', 'all'))
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--out', type=Path, help='Separate calibration directory; defaults to a sibling named for the context limit.')
    parser.add_argument('--model', default='qwen3:8b')
    parser.add_argument('--base-url', default='http://localhost:11434')
    parser.add_argument('--num-ctx', type=int, default=16384)
    parser.add_argument('--max-tokens', type=int, default=8192)
    args = parser.parse_args()
    source = args.run.resolve()
    run = args.out.resolve() if args.out else source.with_name(source.name + f'-calibration-v3-ctx{args.num_ctx}')
    if args.phase in ('judge', 'all'):
        prepare_output(source, run)
        for arm in ('original', 'structural'):
            subprocess.run([sys.executable, str(ROOT / 'scripts/judge_answers.py'), 'judge', '--run', str(run / arm),
                            '--rubric', 'v3', '--model', args.model, '--base-url', args.base_url,
                            '--no-think', '--num-ctx', str(args.num_ctx), '--max-tokens', str(args.max_tokens)],
                           cwd=ROOT, check=True)
    if args.phase in ('report', 'all'):
        report(run)


if __name__ == '__main__':
    main()

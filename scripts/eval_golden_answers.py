#!/usr/bin/env python
"""Paired original/structural golden-answer experiment with frozen inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'data/eval/reports/chat_pipeline/structural-golden-80-2026-10-03'
DOMAINS = ('criminal', 'civil', 'administrative', 'ip')
ARMS = ('original', 'structural')
BASE = ROOT / 'data/eval/reports/chat_pipeline/formal-2026-09-28-articles/answer_pairs.jsonl'
SOURCE_FILES = ('scripts/eval_golden_answers.py', 'src/legal_rag/rag/generator.py',
                'src/legal_rag/rag/context.py', 'src/legal_rag/llm/prompts/answer_system.md',
                'src/legal_rag/verification/policy.py',
                'src/legal_rag/evaluation/answer_replay.py')


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def immutable(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'Frozen manifest changed: {path}')
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def settings():
    from legal_rag.cli import settings_from_options
    return settings_from_options(profile='structural-all', root=ROOT)


def prepare(out):
    from legal_rag.factory import build_hybrid_retriever, close_legs
    from legal_rag.rag.context import ContextBuilder
    from legal_rag.schemas import EvidenceBundle
    selected = [{k: r[k] for k in ('key', 'dataset', 'question', 'reference', 'gold')} for r in rows(BASE)]
    if len(selected) != 80 or len({r['key'] for r in selected}) != 80:
        raise ValueError('Expected 80 unique historical questions')
    out.mkdir(parents=True, exist_ok=True)
    immutable(out / 'selection.json', {'questions': selected, 'reference_sha256': sha(BASE),
              'selection': 'all historical80; development sample; no dedicated IP group',
              'neighbors': 0, 'rendered_context_ceiling': 7500,
              'source_hashes': {p: sha(ROOT / p) for p in SOURCE_FILES}})
    base = settings()
    builder = ContextBuilder(max_chars=1500, max_context_chars=7500, neighbor_radius=0)
    for arm in ARMS:
        directory = out / arm
        directory.mkdir(exist_ok=True)
        versions = [f'{d}-v1' if arm == 'original' else f'{d}-struct800-v1' for d in DOMAINS]
        config = base.model_copy(update={'retrieval': base.retrieval.model_copy(update={'corpora': versions})})
        immutable(directory / 'retrieval.json', {
            'retrieval': config.retrieval.model_dump(mode='json'),
            'embedding': config.embedding.model_dump(mode='json'),
            'rerank': config.rerank.model_dump(mode='json'),
            'indexes': {v: sha(ROOT / 'data/indexes' / v / 'index.json') for v in versions},
            'query_policy': 'raw question; no rewriting or hard filters', 'k_final': 5})
        path = directory / 'contexts.jsonl'
        previous = {r['key']: r for r in rows(path)}
        for ref in selected:
            if ref['key'] in previous:
                row = previous[ref['key']]
                if row['question'] != ref['question']:
                    raise ValueError('Cached question differs')
                if len(EvidenceBundle.model_validate(row['evidence']).render()) > 7500:
                    raise ValueError('Cached evidence exceeds budget')
        todo = [r for r in selected if r['key'] not in previous]
        if todo:
            retriever = build_hybrid_retriever(config)
            try:
                with path.open('a') as handle:
                    for n, ref in enumerate(todo, 1):
                        hits = retriever.retrieve(ref['question'], 5, None)
                        bundle = builder.build(hits)
                        row = {'key': ref['key'], 'question': ref['question'],
                               'evidence': bundle.model_dump(mode='json'),
                               'rendered_chars': len(bundle.render()),
                               'top5': [h.model_dump(mode='json') for h in hits],
                               'trace': retriever.last_trace}
                        handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                        handle.flush()
                        print(f'retrieval {arm} {len(previous)+n}/80', flush=True)
            finally:
                close_legs(retriever)
        if set(r['key'] for r in rows(path)) != set(r['key'] for r in selected):
            raise ValueError('Incomplete or unexpected contexts')
        immutable(directory / 'contexts-frozen.json', {'sha256': sha(path), 'pairs': 80})


def generate(out):
    from legal_rag.evaluation.answers.answer_replay import AnswerReplay
    from legal_rag.schemas import EvidenceBundle
    config = settings()
    if config.llm.backend != 'gemini':
        raise ValueError('Select Gemini in .env')
    selected = json.loads((out / 'selection.json').read_text())
    for path, digest in selected['source_hashes'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('Generation source changed: ' + path)
    replay = AnswerReplay(config, request_interval=15.0)
    contexts, done = {}, {}
    for arm in ARMS:
        d = out / arm
        digest = json.loads((d / 'contexts-frozen.json').read_text())['sha256']
        if sha(d / 'contexts.jsonl') != digest:
            raise ValueError('Frozen contexts changed')
        contexts[arm] = {r['key']: r for r in rows(d / 'contexts.jsonl')}
        done[arm] = {r['key']: r for r in rows(d / 'verify.jsonl')}
        if done[arm] and not (d / 'generation.json').exists():
            raise ValueError('Existing answers lack generation provenance')
        immutable(d / 'generation.json', {'model': config.llm.model, 'temperature': config.llm.temperature,
            'thinking_level': config.llm.gemini_thinking_level, 'max_tokens': config.llm.max_tokens,
            'max_regenerations': config.rag.max_regenerations, 'minimum_interval_seconds': 15,
            'contexts_sha256': digest, 'arm': arm, 'reference_sent_to_generator': False,
            'scoring_text_policy': 'full final answer; no truncation', 'context_budget': 7500})

    def save_pairs(arm):
        pairs = []
        for ref in selected['questions']:
            result = done[arm].get(ref['key'])
            if result is None:
                continue
            bundle = EvidenceBundle.model_validate(contexts[arm][ref['key']]['evidence'])
            docs = [item.document_id for item in bundle.items]
            pairs.append({**ref, 'kind': result['kind'], 'answer': result.get('full_text', ''),
                          'evidence_documents': docs, 'gold_in_evidence': bool(set(docs) & set(ref['gold']))})
        (out / arm / 'answer_pairs.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in pairs))

    errors = 0
    for i, ref in enumerate(selected['questions']):
        for arm in (ARMS if i % 2 == 0 else ARMS[::-1]):
            if done[arm].get(ref['key'], {}).get('kind') in ('answer', 'rejected'):
                continue
            row = contexts[arm][ref['key']]
            result = replay.generate(row)
            bundle = EvidenceBundle.model_validate(row['evidence'])
            result['evidence'] = [' '.join(map(str, item.header.values())) + ' ' + item.text for item in bundle.items]
            result['final_text'] = result.get('full_text', '')
            result['stored_text_truncated'] = False
            with (out / arm / 'verify.jsonl').open('a') as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + '\n')
            done[arm][ref['key']] = result
            save_pairs(arm)
            print(f'generation {i+1}/80 {arm}: {result["kind"]}', flush=True)
            errors = errors + 1 if result['kind'] == 'error' else 0
            if errors >= 3:
                raise RuntimeError('Stopped after 3 API errors; results saved for resume')
    for arm in ARMS:
        save_pairs(arm)


def judge(out):
    for arm in ARMS:
        pairs = rows(out / arm / 'answer_pairs.jsonl')
        if len(pairs) != 80 or any(r['kind'] == 'error' for r in pairs):
            raise ValueError('Complete generation before grading')
    for arm in ARMS:
        subprocess.run([sys.executable, str(ROOT / 'scripts/judge_answers.py'), 'judge', '--run', str(out / arm),
                        '--rubric', 'v2', '--no-think', '--num-ctx', '16384'], check=True, cwd=ROOT)


def report(out):
    from legal_rag.evaluation.answers.answer_reporting import write_golden_report
    write_golden_report(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'generate', 'judge', 'report', 'all'))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    for phase in (('prepare', 'generate', 'judge', 'report') if args.phase == 'all' else (args.phase,)):
        globals()[phase](args.out.resolve())


if __name__ == '__main__':
    main()

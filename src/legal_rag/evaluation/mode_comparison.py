"""Compare finished diagnostic runs of several retrieval modes over one target.

Reads each run's ``target.jsonl``, ``evaluated.jsonl`` and ``summary.json``; nothing is
retrieved again. Runs must share the exact target and settings hash, otherwise a rank
difference could come from a different question set or index rather than the mode.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

RECALL_KS = (5, 10, 20, 50, 100)


def _jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _mode_metrics(ranks: list[int | None], *, top_k: int, deep_k: int) -> dict:
    n = len(ranks)
    found = [rank for rank in ranks if rank is not None]
    buckets = {f'top_{top_k}': sum(rank <= top_k for rank in found),
               f'rank_{top_k + 1}_{deep_k}': sum(top_k < rank <= deep_k for rank in found),
               'not_found': n - len(found)}
    return {
        # Recall beyond the search depth would be a claim the run cannot support.
        'recall_at': {str(k): sum(rank <= k for rank in found) / n for k in RECALL_KS if k <= deep_k},
        'mrr': sum(1 / rank for rank in found) / n,
        'buckets': buckets,
        'bucket_percentages': {key: value * 100 / n for key, value in buckets.items()},
    }


def compare_mode_runs(run_dirs: list[Path]) -> dict:
    runs = []
    for directory in map(Path, run_dirs):
        runs.append({'dir': directory,
                     'summary': json.loads((directory / 'summary.json').read_text(encoding='utf-8')),
                     'target': _jsonl(directory / 'target.jsonl'),
                     'evaluated': {row['case_id']: row for row in _jsonl(directory / 'evaluated.jsonl')}})
    if len(runs) < 2:
        raise ValueError('need at least two runs to compare')
    first = runs[0]
    for run in runs[1:]:
        if run['target'] != first['target']:
            raise ValueError(f"{run['dir']}: target differs from {first['dir']}")
        for key in ('settings_hash', 'top_k', 'deep_k', 'legal_domain', 'corpus_version'):
            if run['summary'].get(key) != first['summary'].get(key):
                label = 'settings' if key == 'settings_hash' else key
                raise ValueError(f"{run['dir']}: {label} {run['summary'].get(key)!r} differs from "
                                 f"{first['summary'].get(key)!r} in {first['dir']}")
    modes = [run['summary']['retrieval_mode'] for run in runs]
    if len(set(modes)) != len(modes):
        raise ValueError(f'each run must be a different retrieval mode, got {modes}')
    case_ids = [row['id'] for row in first['target']]
    for run in runs:
        if set(run['evaluated']) != set(case_ids):
            raise ValueError(f"{run['dir']}: evaluated cases do not match its target")
    # Missing-corpus golds were never searched in any mode; they stay out of every denominator.
    missing = {case_id for run in runs for case_id, row in run['evaluated'].items()
               if row['status'] == 'missing_gold_corpus'}
    scored = [case_id for case_id in case_ids if case_id not in missing]
    if not scored:
        raise ValueError('no evaluable cases')
    top_k, deep_k = first['summary']['top_k'], first['summary']['deep_k']
    summary = first['summary']
    return {
        'corpus_version': summary.get('corpus_version'), 'legal_domain': summary.get('legal_domain'),
        'settings_hash': summary.get('settings_hash'), 'retrieval_corpora': summary.get('retrieval_corpora'),
        'reranker': summary.get('configured_reranker'), 'top_k': top_k, 'deep_k': deep_k,
        'rank_unit': 'retrieved_chunk', 'target_cases': len(scored), 'missing_gold_corpus': len(missing),
        'runs': {mode: str(run['dir']) for mode, run in zip(modes, runs)},
        'modes': {mode: _mode_metrics([run['evaluated'][case_id]['gold_rank'] for case_id in scored],
                                      top_k=top_k, deep_k=deep_k)
                  for mode, run in zip(modes, runs)},
        'cases': [{'case_id': row['id'], 'query': row['query'], 'gold_document_ids': row['relevant_document_ids'],
                   **{mode: run['evaluated'][row['id']]['gold_rank'] for mode, run in zip(modes, runs)}}
                  for row in first['target'] if row['id'] not in missing],
    }


def _rank(rank: int | None) -> str:
    return '—' if rank is None else str(rank)


def comparison_markdown(report: dict) -> str:
    modes = list(report['modes'])
    top_k, deep_k = report['top_k'], report['deep_k']
    header = '| Metric | ' + ' | '.join(modes) + ' |\n|---|' + '---:|' * len(modes) + '\n'
    rows = []
    for k in report['modes'][modes[0]]['recall_at']:
        rows.append(f'| Recall@{k} | ' + ' | '.join(f"{report['modes'][m]['recall_at'][k]:.3f}" for m in modes) + ' |')
    rows.append(f'| MRR@{deep_k} | ' + ' | '.join(f"{report['modes'][m]['mrr']:.4f}" for m in modes) + ' |')
    labels = {f'top_{top_k}': f'Top {top_k}', f'rank_{top_k + 1}_{deep_k}': f'Rank {top_k + 1}–{deep_k}',
              'not_found': 'Not found'}
    for key, label in labels.items():
        rows.append(f'| {label} | ' + ' | '.join(
            f"{report['modes'][m]['buckets'][key]} ({report['modes'][m]['bucket_percentages'][key]:.1f}%)"
            for m in modes) + ' |')
    cases = ['| Case | Question | Gold | ' + ' | '.join(modes) + ' |', '|---|---|---|' + '---:|' * len(modes)]
    for case in report['cases']:
        question = ' '.join(case['query'].split()).replace('|', '\\|')
        question = question if len(question) <= 60 else question[:59] + '…'
        cases.append(f"| `{case['case_id']}` | {question} | `{', '.join(case['gold_document_ids'])}` | "
                     + ' | '.join(_rank(case[m]) for m in modes) + ' |')
    return (
        f"# Retrieval mode comparison — {report['corpus_version']} / {report['legal_domain']}\n\n"
        f"- Target: {report['target_cases']} cases (excluded, gold missing from corpus: {report['missing_gold_corpus']})\n"
        f"- Searched corpora: {', '.join(report['retrieval_corpora'] or [])}\n"
        f"- Reranker: {report['reranker']}; settings hash `{report['settings_hash']}`\n"
        f"- Rank = first retrieved **chunk** rank of the exact gold document; — = not within {deep_k}.\n\n"
        '## Metrics\n\n' + header + '\n'.join(rows) + '\n\n'
        '## Gold rank per case\n\n' + '\n'.join(cases) + '\n'
    )


def write_mode_comparison(output: Path, report: dict) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'comparison.md').write_text(comparison_markdown(report), encoding='utf-8')
    modes = list(report['modes'])
    with (output / 'cases.csv').open('x', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['case_id', 'query', 'gold_document_ids', *(f'{mode}_rank' for mode in modes)])
        for case in report['cases']:
            writer.writerow([case['case_id'], case['query'], json.dumps(case['gold_document_ids'], ensure_ascii=False),
                             *('' if case[mode] is None else case[mode] for mode in modes)])
    return output

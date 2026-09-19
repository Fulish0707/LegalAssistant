"""Diagnostic orchestration over processed QA/chunks and existing retrieval services."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from typing import Callable

from legal_rag.config import Settings
from legal_rag.factory import (
    build_hybrid_retriever, check_index_compatibility, close_legs,
    corpus_settings, select_legs,
)
from legal_rag.ingestion.corpus import Corpus

from .diagnostics import build_duplicate_question_index, diagnose_case, normalize_question
from .qa_dataset import _rows
from .reports import summarize_cases, write_diagnostic_reports

_EVALUATION_SPLITS = {'test', 'eval', 'evaluation', 'validation', 'valid', 'val', 'dev'}


def _split(row: dict) -> str:
    return str(row.get('split') or row.get('metadata', {}).get('split') or '').strip().lower()


def load_diagnostic_rows(qa_path: Path, *, domain: str, dataset: Path | None = None) -> tuple[list[dict], dict, dict]:
    """Retain QA rows and full-domain exact-question positives; never index answers."""
    rows = []
    other_domain_pairs = set()
    for line, row in _rows(Path(qa_path)):
        if row.get('legal_domain') != domain:
            if all(isinstance(row.get(key), str) and row[key].strip() for key in ('question', 'document_id')):
                other_domain_pairs.add((normalize_question(row['question']), row['document_id']))
            continue
        if not all(isinstance(row.get(key), str) and row[key].strip() for key in ('question', 'document_id')):
            raise ValueError(f'{qa_path}:{line}: question and document_id must be nonempty strings')
        # Source line distinguishes repeated question/document rows without merging labels.
        identity = json.dumps([domain, row['document_id'], row['question'], line], ensure_ascii=False)
        rows.append({**row, 'case_id': 'qa:' + hashlib.sha256(identity.encode()).hexdigest()[:20]})
    if not rows:
        raise ValueError(f'{qa_path}: no QA records for domain {domain!r}')
    duplicates = build_duplicate_question_index(rows)
    provenance = {'qa_file': str(Path(qa_path).resolve()), 'domain_qa_rows': len(rows)}
    has_splits = any(_split(row) for row in rows)
    domain_pairs = {(normalize_question(row['question']), row['document_id']) for row in rows}
    if has_splits:
        rows = [row for row in rows if _split(row) in _EVALUATION_SPLITS]
    if dataset is not None:
        pairs = set()
        for line, example in _rows(Path(dataset)):
            question, golds = example.get('query'), example.get('relevant_document_ids')
            if not isinstance(question, str) or not question.strip() or not isinstance(golds, list) or not golds:
                raise ValueError(f'{dataset}:{line}: expected retrieval evaluation query and relevant_document_ids')
            if not all(isinstance(gold, str) and gold for gold in golds):
                raise ValueError(f'{dataset}:{line}: invalid relevant_document_ids')
            pairs.update((normalize_question(question), gold) for gold in golds)
        # Ignore known other-domain pairs, but never hide missing target QA labels.
        pairs -= other_domain_pairs - domain_pairs
        selected = [row for row in rows if (normalize_question(row['question']), row['document_id']) in pairs]
        matched = {(normalize_question(row['question']), row['document_id']) for row in selected}
        if pairs - matched:
            raise ValueError(f'{dataset}: {len(pairs - matched)} evaluation question/gold pairs do not match QA domain {domain!r}')
        rows = selected
        provenance.update(selection='evaluation_dataset', dataset=str(Path(dataset).resolve()))
    elif has_splits:
        provenance['selection'] = 'qa_evaluation_split'
    else:
        provenance['selection'] = 'unsplit_processed_qa'
    if not rows:
        raise ValueError('no evaluation rows selected; training-only QA is not an evaluation split')
    return sorted(rows, key=lambda row: row['case_id']), duplicates, provenance


def _eligible(case: dict, only_failures: bool) -> bool:
    return case['status'] != 'missing_gold_corpus' and (not only_failures or not case['gold_found_in_top_k'])


def _reservoir_add(selected: list, case: dict, seen: int, limit: int, rng: random.Random) -> None:
    if len(selected) < limit:
        selected.append(case)
    else:
        slot = rng.randrange(seen)
        if slot < limit:
            selected[slot] = case


def select_review_cases(cases: list[dict], *, limit: int, seed: int, only_failures: bool) -> list[dict]:
    if limit < 1:
        raise ValueError('limit must be positive')
    selected, seen, rng = [], 0, random.Random(seed)
    for case in sorted(cases, key=lambda case: case['case_id']):
        if _eligible(case, only_failures):
            seen += 1
            _reservoir_add(selected, case, seen, limit, rng)
    return sorted(selected, key=lambda case: case['case_id'])


def validate_index_metadata(settings: Settings, metadata: dict, *, collection: str, dimension: int) -> None:
    expected = {'corpus_version': settings.corpus.version, 'collection': collection, 'dimension': dimension}
    errors = [f'{key}: recorded {metadata.get(key)!r}, expected {value!r}'
              for key, value in expected.items() if metadata.get(key) != value]
    if errors:
        raise ValueError('incompatible index metadata: ' + '; '.join(errors))


def _qa_path(settings: Settings, supplied: Path | None) -> Path:
    if supplied is not None:
        return Path(supplied)
    directory = settings.processed_corpus_dir()
    local = directory / 'qa.jsonl'
    if local.is_file():
        return local
    # Imported corpora retain their processed source paths in the manifest.
    manifest = directory / 'manifest.json'
    candidates = set()
    if manifest.is_file():
        for source in json.loads(manifest.read_text(encoding='utf-8')).get('raw_files', []):
            candidate = Path(source['file']).parent / 'qa.jsonl'
            if candidate.is_file():
                candidates.add(candidate.resolve())
    if len(candidates) == 1:
        return candidates.pop()
    raise FileNotFoundError(f'no unambiguous processed qa.jsonl at {directory}; provide --qa-file')


def _gold_chunk_index(corpus: Corpus, document_ids: set[str], *, max_gold_chunks: int, preview_chars: int) -> tuple[dict, Counter]:
    """Stream a potentially gigabyte corpus; retain bounded previews only for target golds."""
    previews, counts = defaultdict(list), Counter()
    total = 0
    for chunk in corpus.chunks():
        total += 1
        if chunk.document_id not in document_ids:
            continue
        counts[chunk.document_id] += 1
        bucket = previews[chunk.document_id]
        bucket.append(chunk.model_copy(update={'text': chunk.text[:preview_chars]}))
        bucket.sort(key=lambda item: (item.order, item.id))
        del bucket[max_gold_chunks:]
    if not total:
        raise ValueError(f'no processed chunks at {corpus.directory}')
    return previews, counts


def gold_chunks_across_corpora(corpora: dict[str, Corpus], document_ids: set[str], *,
                               max_gold_chunks: int, preview_chars: int) -> dict[str, list]:
    """Gold chunks for documents that may live in any of several processed corpora.

    A fan-out target draws its cases from more than one corpus, so no single corpus holds every
    gold document. A document found in two corpora is an ambiguous gold and raises.
    """
    found: dict[str, list] = {}
    owner: dict[str, str] = {}
    for version, corpus in corpora.items():
        previews, _ = _gold_chunk_index(corpus, document_ids, max_gold_chunks=max_gold_chunks,
                                        preview_chars=preview_chars)
        for document_id, chunks in previews.items():
            if document_id in owner:
                raise ValueError(f'gold document {document_id} has chunks in both {owner[document_id]} '
                                 f'and {version}; the corpora overlap and the gold is ambiguous')
            owner[document_id] = version
            found[document_id] = chunks
    return found


def deepen_retrieval(settings: Settings, deep_k: int) -> Settings:
    """Raise candidate depths on a copy; fusion weights, models and reranker are unchanged."""
    return settings.model_copy(update={'retrieval': settings.retrieval.model_copy(update={
        'k_dense': max(deep_k, settings.retrieval.k_dense),
        'k_sparse': max(deep_k, settings.retrieval.k_sparse),
        'k_fused': max(deep_k, settings.retrieval.k_fused),
    })})


def run_retrieval_diagnostics(
    settings: Settings, *, domain: str, retrieval_mode: str = 'hybrid', limit: int = 50,
    top_k: int = 20, deep_k: int = 100, seed: int = 42, only_failures: bool = False,
    output: Path | None = None, qa_file: Path | None = None, dataset: Path | None = None,
    preview_chars: int = 1200, max_gold_chunks: int = 10,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Evaluate the entire target, then reservoir-sample review cases without retaining all hits."""
    if retrieval_mode not in {'dense', 'sparse', 'hybrid'}:
        raise ValueError('retrieval_mode must be dense, sparse, or hybrid')
    if not 1 <= top_k <= deep_k or min(limit, preview_chars, max_gold_chunks) < 1:
        raise ValueError('require 1 <= top_k <= deep_k and positive limits')
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = Path(output) if output else settings.resolve(Path('reports/retrieval_diagnostics')) / settings.corpus.version / domain / f'{retrieval_mode}-{timestamp}'
    if output.exists():
        raise FileExistsError(f'report directory already exists; choose a new --output: {output}')
    corpus = Corpus.load(settings.processed_corpus_dir())
    if not (corpus.directory / 'chunks.jsonl').is_file():
        raise FileNotFoundError(f'no processed chunks at {corpus.directory}')
    qa_file = _qa_path(settings, qa_file)
    if dataset is None:
        default_dataset = settings.resolve(settings.paths.eval_dir) / f'{settings.corpus.version}.jsonl'
        dataset = default_dataset if default_dataset.is_file() else None
    rows, duplicates, provenance = load_diagnostic_rows(qa_file, domain=domain, dataset=dataset)
    if progress:
        progress(f'Target: {len(rows)} QA rows ({provenance["selection"]}); scanning gold chunks')
    previews, counts = _gold_chunk_index(corpus, {row['document_id'] for row in rows},
                                        max_gold_chunks=max_gold_chunks, preview_chars=preview_chars)
    # Change only candidate depth on a copied configuration. Fusion weights, models,
    # tokenization and the configured reranker are unchanged.
    scoped = deepen_retrieval(settings, deep_k)
    versions = scoped.retrieval.corpora or [scoped.corpus.version]
    if scoped.corpus.version not in versions:
        raise ValueError('target QA corpus_version is not among the configured retrieval corpora')
    # Check files before opening any stores (factory otherwise may partially open).
    metadata = {version: check_index_compatibility(corpus_settings(scoped, version)) for version in versions}
    for version in versions:
        item = corpus_settings(scoped, version)
        if item.qdrant.mode == 'embedded' and not item.qdrant_path().is_dir():
            raise FileNotFoundError(f'no existing Qdrant index at {item.qdrant_path()}')
    hybrid = build_hybrid_retriever(scoped)
    try:
        stores = {leg.name.rsplit(':', 1)[0]: leg.store for leg in hybrid.legs}
        for version, store in stores.items():
            validate_index_metadata(corpus_settings(scoped, version), metadata[version],
                                    collection=store.collection, dimension=store.dimension)
            if not store.exists():
                raise ValueError(f'Qdrant collection does not exist: {store.collection}')
        retriever = select_legs(hybrid, retrieval_mode)
        selected, metrics, missing = [], [], []
        eligible, rng = 0, random.Random(seed)
        for number, row in enumerate(rows, 1):
            duplicate = duplicates[normalize_question(row['question'])]
            case = diagnose_case(row, retriever, gold_chunks=previews.get(row['document_id'], []),
                                 gold_chunk_count=counts[row['document_id']],
                                 duplicate_question_count=duplicate['count'],
                                 candidate_gold_document_ids=duplicate['document_ids'],
                                 top_k=top_k, deep_k=deep_k, preview_chars=preview_chars,
                                 max_gold_chunks=max_gold_chunks, case_id=row['case_id'])
            metrics.append({key: case[key] for key in ('case_id', 'status', 'gold_rank', 'multi_positive_hit',
                                                        'gold_found_in_top_k', 'duplicate_question_count')})
            if case['status'] == 'missing_gold_corpus':
                missing.append(case)
            elif _eligible(case, only_failures):
                eligible += 1
                _reservoir_add(selected, case, eligible, limit, rng)
            if progress and (number % 10 == 0 or number == len(rows)):
                progress(f'Evaluated {number}/{len(rows)}; eligible review cases {eligible}')
        summary = summarize_cases(metrics, legal_domain=domain, retrieval_mode=retrieval_mode,
                                  top_k=top_k, deep_k=deep_k)
        summary.update(corpus_version=settings.corpus.version, seed=seed, limit=limit,
                       only_failures=only_failures, report_cases=len(selected), eligible_review_cases=eligible,
                       source=provenance, qa_sha256=_sha256(qa_file),
                       dataset_sha256=_sha256(dataset) if dataset else None,
                       evaluated_case_ids=[row['case_id'] for row in rows],
                       selected_case_ids=sorted(case['case_id'] for case in selected),
                       retrieval_corpora=versions, index_metadata=metadata,
                       candidate_depths={'per_leg': hybrid.k_leg, 'fused': hybrid.k_fused},
                       configured_reranker=settings.rerank.backend,
                       filters={'legal_domain': domain},
                       preview_chars=preview_chars, max_gold_chunks=max_gold_chunks,
                       settings_hash=scoped.hash())
        write_diagnostic_reports(output, sorted(selected, key=lambda c: c['case_id']), missing, summary)
        # Compact per-case metrics support paired mode comparisons, including successes.
        with (output / 'evaluated.jsonl').open('x', encoding='utf-8') as handle:
            for row in metrics:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        with (output / 'target.jsonl').open('x', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps({'id': row['case_id'], 'query': row['question'],
                                         'relevant_document_ids': [row['document_id']]}, ensure_ascii=False) + '\n')
        return output
    finally:
        close_legs(hybrid)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

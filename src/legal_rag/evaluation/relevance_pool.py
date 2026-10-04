"""Pool retrieved chunks from several modes for relevance judging, and record rankings."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Callable

from legal_rag.retrieval.hybrid import Retriever
from legal_rag.schemas import Chunk, RetrievalFilters

from .case_labels import case_label


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _pair(case: dict, chunk: Chunk) -> dict:
    fields = dict(document_id=chunk.document_id, document_type=str(chunk.document_type), title=chunk.title,
                  text=chunk.text, authority=chunk.metadata.authority or chunk.metadata.court,
                  date=chunk.metadata.original.get('date'))
    return dict(case_id=case['case_id'], chunk_id=chunk.id, question=case['question'],
                question_domain=case.get('legal_domain'),
                reference_answer=case.get('answer'), **{**fields, **case_label(fields)},
                ranks={}, is_gold_chunk=chunk.document_id == case['document_id'])


def domain_filters(case: dict) -> RetrievalFilters:
    """The default: each case is retrieved inside its own legal domain."""
    return RetrievalFilters(legal_domain=case['legal_domain'])


def no_filters(case: dict) -> None:
    """Unfiltered retrieval, as a multi-corpus fan-out is served."""
    return None


def _retriever_for(mode: Retriever | Callable[[dict], Retriever], case: dict) -> Retriever:
    return mode if hasattr(mode, 'retrieve') else mode(case)


def build_pool(cases: list[dict], retrievers: dict[str, Retriever | Callable[[dict], Retriever]], *,
               gold_chunks: dict[str, list[Chunk]], pool_depth: int, run_depth: int,
               progress: Callable[[str], None] | None = None,
               case_filters: Callable[[dict], RetrievalFilters | None] = domain_filters,
               ) -> tuple[list[dict], dict[str, list[dict]]]:
    """Retrieve every case in every mode, pooling the top ``pool_depth`` of each for judging.

    A mode's value is a retriever, or a callable returning the retriever for a case, which is how
    a run can query only that case's own corpus. ``case_filters`` returns the filters a case is
    retrieved under, or None for unfiltered retrieval.
    """
    if not 1 <= pool_depth <= run_depth:
        raise ValueError('require 1 <= pool_depth <= run_depth')
    pool: dict[tuple[str, str], dict] = {}
    runs: dict[str, list[dict]] = {mode: [] for mode in retrievers}
    for index, case in enumerate(cases, 1):
        filters = case_filters(case)
        for mode, chosen in retrievers.items():
            retriever = _retriever_for(chosen, case)
            hits = retriever.retrieve(case['question'], run_depth, filters)[:run_depth]
            runs[mode].append({'case_id': case['case_id'], 'ranked': [
                {'chunk_id': hit.chunk.id, 'document_id': hit.chunk.document_id} for hit in hits]})
            for rank, hit in enumerate(hits[:pool_depth], 1):
                row = pool.setdefault((case['case_id'], hit.chunk.id), _pair(case, hit.chunk))
                row['ranks'][mode] = rank
        # Gold chunks are judged whatever their rank, to check the judge against the source.
        for chunk in gold_chunks.get(case['document_id'], []):
            pool.setdefault((case['case_id'], chunk.id), _pair(case, chunk))
        if progress and (index % 10 == 0 or index == len(cases)):
            progress(f'Pooled {index}/{len(cases)} cases; {len(pool)} pairs')
    return list(pool.values()), runs


def adopt_run(pool: list[dict], runs: dict[str, list[dict]], name: str, rows: list[dict],
              source_pool: list[dict], *, pool_depth: int) -> tuple[list[dict], dict[str, list[dict]]]:
    """Add a run recorded elsewhere over the same cases, with the pool rows it needs.

    A copied run only measures correctly when every pair it ranks within ``pool_depth`` has a pool
    row: ``graded_metrics.restrict_to_pool`` drops judgments for unpooled pairs, which would score
    the copied run's hits as grade 0. Rows missing from ``pool`` are taken from ``source_pool``
    (the pool directory the run was copied from); a pair in neither raises. Inputs are not mutated.
    """
    if name in runs:
        raise ValueError(f'run {name!r} is already recorded')
    merged = [dict(row, ranks=dict(row['ranks'])) for row in pool]
    by_pair = {(row['case_id'], row['chunk_id']): row for row in merged}
    source = {(row['case_id'], row['chunk_id']): row for row in source_pool}
    added: list[dict] = []
    for row in rows:
        for rank, hit in enumerate(row['ranked'][:pool_depth], 1):
            pair = (row['case_id'], hit['chunk_id'])
            adopted = by_pair.get(pair)
            if adopted is None:
                origin = source.get(pair)
                if origin is None:
                    raise ValueError(f'{pair} is ranked by {name!r} but is in neither pool; its judgment '
                                     'would be dropped and the hit scored as irrelevant')
                adopted = dict(origin, ranks={})
                by_pair[pair] = adopted
                added.append(adopted)
            adopted['ranks'][name] = rank
    return merged + added, {**runs, name: rows}


def write_pool(output: Path, pool: list[dict], runs: dict[str, list[dict]], provenance: dict) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'runs').mkdir()
    for name, rows in [('pool.jsonl', pool), *((f'runs/{mode}.jsonl', rows) for mode, rows in runs.items())]:
        with (output / name).open('x', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    (output / 'pool.json').write_text(json.dumps(
        {**provenance, 'pairs': len(pool), 'modes': list(runs)}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return output


def derive_pool(pool_rows: list[dict], runs: dict[str, list[dict]], *, pool_depth: int,
                include_unretrieved_gold: bool) -> list[dict]:
    """Rows of a shallower pool, taken from an existing pool without retrieving again.

    Ranks are recomputed from the runs to ``pool_depth`` and must agree with the source rows'
    ranks, so a depth the source pool never reached (or a mismatched runs directory) raises.
    Gold chunks not retrieved within the depth are kept, with empty ranks, only when
    ``include_unretrieved_gold``. Source order and all other fields are kept.
    """
    if pool_depth < 1:
        raise ValueError('pool_depth must be >= 1')
    ranks: dict[tuple[str, str], dict[str, int]] = {}
    for mode, rows in runs.items():
        for row in rows:
            for rank, hit in enumerate(row['ranked'][:pool_depth], 1):
                ranks.setdefault((row['case_id'], hit['chunk_id']), {})[mode] = rank
    by_pair = {(row['case_id'], row['chunk_id']): row for row in pool_rows}
    missing = [pair for pair in ranks if pair not in by_pair]
    if missing:
        raise ValueError(f'{len(missing)} pairs ranked within depth {pool_depth} are not in the source pool '
                         f'(e.g. {missing[0]}); the source pool is shallower or from other runs')
    derived = []
    for row in pool_rows:
        pair = (row['case_id'], row['chunk_id'])
        trimmed = {mode: rank for mode, rank in row['ranks'].items() if rank <= pool_depth}
        if trimmed != ranks.get(pair, {}):
            raise ValueError(f'pool ranks {row["ranks"]} for {pair} disagree with the runs')
        if trimmed or (row['is_gold_chunk'] and include_unretrieved_gold):
            derived.append(dict(row, ranks=trimmed))
    return derived


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


JUDGMENT_FILES = ('judgments.jsonl', 'judge_runs.jsonl')


def derive_pool_dir(source: Path, output: Path, *, pool_depth: int, include_unretrieved_gold: bool) -> Path:
    """Write a shallower pool directory derived from ``source`` (which is only read).

    runs/ are copied unchanged. judgments.jsonl and judge_runs.jsonl are copied when present, so
    pairs already judged are reused (judgment keys do not depend on the pool directory).
    """
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    provenance = json.loads((source / 'pool.json').read_text(encoding='utf-8'))
    source_depth = int(provenance['pool_depth'])
    if pool_depth > source_depth:
        raise ValueError(f'cannot deepen: pool_depth {pool_depth} > source pool_depth {source_depth}')
    modes = provenance.get('modes') or sorted(path.stem for path in (source / 'runs').glob('*.jsonl'))
    runs = {mode: read_jsonl(source / 'runs' / f'{mode}.jsonl') for mode in modes}
    rows = derive_pool(read_jsonl(source / 'pool.jsonl'), runs, pool_depth=pool_depth,
                       include_unretrieved_gold=include_unretrieved_gold)
    copied = [name for name in JUDGMENT_FILES if (source / name).exists()]
    write_pool(output, rows, runs, {
        **provenance, 'pool_depth': pool_depth, 'include_unretrieved_gold': include_unretrieved_gold,
        'derived_from': {'pool_dir': str(source), 'pool_jsonl_sha256': _sha256(source / 'pool.jsonl'),
                         'pool_depth': source_depth},
        'judgments_copied_from': str(source) if copied else None})
    for name in copied:
        shutil.copyfile(source / name, output / name)
    return output

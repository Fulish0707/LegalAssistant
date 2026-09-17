"""Pool retrieved chunks from several modes for relevance judging, and record rankings."""
from __future__ import annotations

import json
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
                reference_answer=case.get('answer'), **{**fields, **case_label(fields)},
                ranks={}, is_gold_chunk=chunk.document_id == case['document_id'])


def build_pool(cases: list[dict], retrievers: dict[str, Retriever], *, gold_chunks: dict[str, list[Chunk]],
               pool_depth: int, run_depth: int, progress: Callable[[str], None] | None = None,
               ) -> tuple[list[dict], dict[str, list[dict]]]:
    if not 1 <= pool_depth <= run_depth:
        raise ValueError('require 1 <= pool_depth <= run_depth')
    pool: dict[tuple[str, str], dict] = {}
    runs: dict[str, list[dict]] = {mode: [] for mode in retrievers}
    for index, case in enumerate(cases, 1):
        filters = RetrievalFilters(legal_domain=case['legal_domain'])
        for mode, retriever in retrievers.items():
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

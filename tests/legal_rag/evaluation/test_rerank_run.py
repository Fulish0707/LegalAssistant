from __future__ import annotations

import pytest

from legal_rag.evaluation.rerank_run import rerank_run
from legal_rag.schemas import Chunk, ChunkMetadata, DocumentType


def _chunks():
    return [Chunk(id=f'd{i}#p', document_id=f'd{i}', document_type=DocumentType.statute, path='p', order=0,
                  text=f'text {i}', title='t', metadata=ChunkMetadata(dataset='f', corpus_version='v'))
            for i in (1, 2, 3)]


class _Reranker:
    """Sorts by a per-chunk score, the way a cross-encoder would."""

    name = 'fake'

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.queries: list[str] = []

    def rerank(self, query, candidates, k):
        self.queries.append(query)
        ordered = sorted(candidates, key=lambda hit: -self.scores.get(hit.chunk.id, 0.0))
        return [hit.model_copy(update={'retriever': 'reranked'}) for hit in ordered[:k]]


def _fixtures():
    chunks = {chunk.id: chunk for chunk in _chunks()}
    rows = [{'case_id': 'c1', 'ranked': [{'chunk_id': chunk_id, 'document_id': chunk.document_id}
                                         for chunk_id, chunk in chunks.items()]}]
    return chunks, rows


def test_reorders_each_case_by_the_reranker_and_keeps_the_row_shape():
    chunks, rows = _fixtures()
    ids = list(chunks)
    reranker = _Reranker({ids[2]: 3.0, ids[0]: 2.0, ids[1]: 1.0})
    out = rerank_run(rows, chunks, reranker, {'c1': 'question?'}, depth=3)
    assert out == [{'case_id': 'c1', 'ranked': [{'chunk_id': ids[2], 'document_id': chunks[ids[2]].document_id},
                                                {'chunk_id': ids[0], 'document_id': chunks[ids[0]].document_id},
                                                {'chunk_id': ids[1], 'document_id': chunks[ids[1]].document_id}]}]
    assert reranker.queries == ['question?']


def test_truncates_to_depth():
    chunks, rows = _fixtures()
    out = rerank_run(rows, chunks, _Reranker({}), {'c1': 'q'}, depth=1)
    assert len(out[0]['ranked']) == 1


def test_a_missing_chunk_or_question_raises():
    chunks, rows = _fixtures()
    with pytest.raises(ValueError, match='not in the processed corpus'):
        rerank_run(rows, {}, _Reranker({}), {'c1': 'q'}, depth=3)
    with pytest.raises(ValueError, match='no question'):
        rerank_run(rows, chunks, _Reranker({}), {}, depth=3)


def test_reports_progress_per_case():
    chunks, rows = _fixtures()
    seen: list[str] = []
    rerank_run(rows, chunks, _Reranker({}), {'c1': 'q'}, depth=3, progress=seen.append)
    assert seen and '1/1' in seen[-1]


def _load_script():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[3] / 'scripts' / 'rerank_relevance_pool.py'
    spec = importlib.util.spec_from_file_location('script_rerank_relevance_pool', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_baseline_map_accepts_plain_names_and_output_equals_source():
    script = _load_script()
    assert script._baseline_map(['dense', 'reranked_512=reranked']) == {
        'dense': 'dense', 'reranked_512': 'reranked'}
    with pytest.raises(Exception, match='NAME or OUTPUT=SOURCE'):
        script._baseline_map(['reranked_512='])


class _Settings:
    class rerank:
        model = 'settings/model'
        device = 'cpu'

    def hash(self):
        return 'h'


def _source_dir(tmp_path):
    """A one-case source pool whose runs rank three chunks, the fused run last-best."""
    from legal_rag.evaluation.relevance_pool import write_pool
    chunks = {chunk.id: chunk for chunk in _chunks()}
    ids = list(chunks)
    ranked = [{'chunk_id': cid, 'document_id': chunks[cid].document_id} for cid in ids]
    runs = {name: [{'case_id': 'c1', 'ranked': ranked}] for name in ('hybrid_s0.75', 'reranked')}
    pool = [{'case_id': 'c1', 'chunk_id': cid, 'document_id': chunks[cid].document_id, 'question': 'q',
             'text': chunks[cid].text, 'title': 't', 'ranks': {'hybrid_s0.75': i}, 'is_gold_chunk': False}
            for i, cid in enumerate(ids, 1)]
    return write_pool(tmp_path / 'src', pool, runs,
                      {'target': str(tmp_path / 'missing.jsonl'), 'target_sha256': 't', 'pool_depth': 3}), chunks


def test_script_passes_max_length_and_batch_size_to_the_reranker_and_renames_a_baseline(tmp_path, monkeypatch):
    import json
    from typer.testing import CliRunner
    script = _load_script()
    source, chunks = _source_dir(tmp_path)
    built = {}

    def _reranker(model, device, batch_size, max_length):
        built.update(model=model, device=device, batch_size=batch_size, max_length=max_length)
        return _Reranker({})

    monkeypatch.setattr(script, 'settings_from_options', lambda **_: _Settings())
    monkeypatch.setattr(script, '_needed_chunks', lambda settings, ids: chunks)
    monkeypatch.setattr(script, 'CrossEncoderReranker',
                        lambda model, device, batch_size, max_length: _reranker(model, device, batch_size, max_length))
    out = tmp_path / 'out'
    result = CliRunner().invoke(script.app, [
        '--pool-dir', str(source), '--output', str(out), '--depth', '3', '--run-depth', '3',
        '--baseline', 'hybrid_s0.75', '--baseline', 'reranked_512=reranked',
        '--run-name', 'reranked_1024', '--max-length', '1024', '--batch-size', '4'])
    assert result.exit_code == 0, result.output
    assert built == {'model': 'settings/model', 'device': 'cpu', 'batch_size': 4, 'max_length': 1024}
    assert (out / 'runs' / 'reranked_512.jsonl').read_bytes() == (source / 'runs' / 'reranked.jsonl').read_bytes()
    assert (out / 'runs' / 'reranked_1024.jsonl').exists() and not (out / 'runs' / 'reranked.jsonl').exists()
    provenance = json.loads((out / 'pool.json').read_text(encoding='utf-8'))
    assert provenance['rerank']['max_length'] == 1024 and provenance['rerank']['batch_size'] == 4
    assert provenance['rerank']['baselines'] == {'hybrid_s0.75': 'hybrid_s0.75', 'reranked_512': 'reranked'}
    assert set(provenance['derived_from']['runs_sha256']) == {'hybrid_s0.75', 'reranked'}   # keyed by source name
    assert sorted(provenance['modes']) == ['hybrid_s0.75', 'reranked_1024', 'reranked_512']


def test_script_refuses_a_run_name_that_collides_with_a_baseline(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    script = _load_script()
    source, chunks = _source_dir(tmp_path)
    monkeypatch.setattr(script, 'settings_from_options', lambda **_: _Settings())
    result = CliRunner().invoke(script.app, [
        '--pool-dir', str(source), '--output', str(tmp_path / 'out2'), '--depth', '3',
        '--baseline', 'reranked', '--run-name', 'reranked'])
    assert result.exit_code != 0 and 'collides' in result.output

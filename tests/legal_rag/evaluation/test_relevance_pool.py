import json

import pytest

from legal_rag.config import Settings
from legal_rag.evaluation.diagnostic_runner import deepen_retrieval
from legal_rag.evaluation.relevance_pool import build_pool, no_filters, read_jsonl, write_pool
from legal_rag.schemas import Chunk, ChunkMetadata, DocumentType, ScoredChunk


def make_chunk(chunk_id, document_id=None, text='본문'):
    return Chunk(id=chunk_id, document_id=document_id or chunk_id.split('#')[0], document_type=DocumentType.judgment,
                 path='p', order=0, text=text, title='제목',
                 metadata=ChunkMetadata(dataset='d', corpus_version='ip-v1', authority='특허법원'))


class ListRetriever:
    def __init__(self, ranking):
        self.ranking, self.calls = ranking, []

    def retrieve(self, query, k, filters=None):
        self.calls.append((query, k, filters))
        return [ScoredChunk(chunk=make_chunk(c), score=1.0, retriever='fused') for c in self.ranking[:k]]


CASES = [dict(case_id='c1', question='질문', answer='답변', document_id='gold', legal_domain='intellectual_property')]


def test_pool_is_union_of_top_depth_plus_gold_with_each_modes_rank():
    retrievers = {'dense': ListRetriever(['a#0', 'b#0', 'c#0']), 'sparse': ListRetriever(['b#0', 'd#0'])}
    pool, runs = build_pool(CASES, retrievers, gold_chunks={'gold': [make_chunk('gold#0')]},
                            pool_depth=2, run_depth=3)
    rows = {row['chunk_id']: row for row in pool}
    assert set(rows) == {'a#0', 'b#0', 'd#0', 'gold#0'}           # c#0 is dense rank 3 > pool_depth
    assert rows['b#0']['ranks'] == {'dense': 2, 'sparse': 1}
    assert rows['gold#0']['ranks'] == {} and rows['gold#0']['is_gold_chunk']
    assert rows['a#0']['question'] == '질문' and rows['a#0']['reference_answer'] == '답변'
    assert rows['a#0']['search_query'] == '특허법원 a'
    assert runs['dense'] == [{'case_id': 'c1', 'ranked': [
        {'chunk_id': 'a#0', 'document_id': 'a'}, {'chunk_id': 'b#0', 'document_id': 'b'},
        {'chunk_id': 'c#0', 'document_id': 'c'}]}]


def test_retrieved_gold_chunk_keeps_its_ranks():
    pool, _ = build_pool(CASES, {'dense': ListRetriever(['x#0', 'gold#0'])},
                         gold_chunks={'gold': [make_chunk('gold#0'), make_chunk('gold#1', 'gold')]},
                         pool_depth=2, run_depth=2)
    rows = {row['chunk_id']: row for row in pool}
    assert rows['gold#0']['ranks'] == {'dense': 2} and rows['gold#0']['is_gold_chunk']
    assert rows['gold#1']['ranks'] == {} and rows['gold#1']['is_gold_chunk']
    assert not rows['x#0']['is_gold_chunk']
    assert len(pool) == 3


def test_retrieval_uses_run_depth_and_the_domain_filter():
    # The default stays the filtered single-mode retrieval that earlier pools were built with.
    dense = ListRetriever(['a#0'])
    build_pool(CASES, {'dense': dense}, gold_chunks={}, pool_depth=1, run_depth=100)
    query, k, filters = dense.calls[0]
    assert (query, k, filters.legal_domain) == ('질문', 100, 'intellectual_property')


def test_unfiltered_retrieval_passes_no_filters():
    dense = ListRetriever(['a#0'])
    build_pool(CASES, {'dense': dense}, gold_chunks={}, pool_depth=1, run_depth=10, case_filters=no_filters)
    assert dense.calls[0][2] is None


def test_a_mode_may_route_to_a_retriever_chosen_per_case():
    cases = [dict(CASES[0], case_id='c1', legal_domain='criminal'),
             dict(CASES[0], case_id='c2', legal_domain='civil')]
    by_domain = {'criminal': ListRetriever(['a#0']), 'civil': ListRetriever(['b#0'])}
    pool, runs = build_pool(cases, {'oracle': lambda case: by_domain[case['legal_domain']]},
                            gold_chunks={}, pool_depth=1, run_depth=1, case_filters=no_filters)
    assert [row['chunk_id'] for row in pool] == ['a#0', 'b#0']
    assert [len(retriever.calls) for retriever in by_domain.values()] == [1, 1]
    assert [row['case_id'] for row in runs['oracle']] == ['c1', 'c2']


def test_pool_depth_cannot_exceed_run_depth():
    with pytest.raises(ValueError):
        build_pool(CASES, {'dense': ListRetriever([])}, gold_chunks={}, pool_depth=20, run_depth=10)


def test_progress_reports_every_ten_cases():
    cases = [dict(CASES[0], case_id=f'c{i}') for i in range(25)]
    messages = []
    build_pool(cases, {'dense': ListRetriever(['a#0'])}, gold_chunks={}, pool_depth=1, run_depth=1,
               progress=messages.append)
    assert len(messages) == 3


def test_write_refuses_existing_directory(tmp_path):
    pool, runs = build_pool(CASES, {'dense': ListRetriever(['a#0'])}, gold_chunks={}, pool_depth=1, run_depth=1)
    write_pool(tmp_path / 'out', pool, runs, {'target_sha256': 'x'})
    assert read_jsonl(tmp_path / 'out' / 'pool.jsonl')[0]['chunk_id'] == 'a#0'
    assert read_jsonl(tmp_path / 'out' / 'runs' / 'dense.jsonl') == runs['dense']
    assert json.loads((tmp_path / 'out' / 'pool.json').read_text())['pairs'] == 1
    with pytest.raises(FileExistsError):
        write_pool(tmp_path / 'out', pool, runs, {})


def test_deepen_retrieval_raises_depths_on_a_copy_only():
    settings = Settings(corpus={'version': 'v1'}, retrieval={'k_dense': 150, 'k_sparse': 20, 'k_fused': 30,
                                                             'rrf_k': 7, 'corpora': ['v1', 'v2']})
    deep = deepen_retrieval(settings, 100)
    assert (deep.retrieval.k_dense, deep.retrieval.k_sparse, deep.retrieval.k_fused) == (150, 100, 100)
    assert deep.retrieval.rrf_k == 7 and deep.retrieval.corpora == ['v1', 'v2']
    assert deep.retrieval.k_final == settings.retrieval.k_final
    assert (settings.retrieval.k_sparse, settings.retrieval.k_fused) == (20, 30)


def _source_pool(tmp_path):
    """Depth-3 pool over runs of depth 5: gold#0 retrieved at dense 2, gold#1 at sparse 3, gold#2 never."""
    retrievers = {'dense': ListRetriever(['a#0', 'gold#0', 'b#0', 'c#0', 'd#0']),
                  'sparse': ListRetriever(['b#0', 'e#0', 'gold#1', 'a#0', 'f#0'])}
    gold = [make_chunk('gold#0'), make_chunk('gold#1', 'gold'), make_chunk('gold#2', 'gold')]
    pool, runs = build_pool(CASES, retrievers, gold_chunks={'gold': gold}, pool_depth=3, run_depth=5)
    source = write_pool(tmp_path / 'src', pool, runs, {'target_sha256': 't', 'pool_depth': 3, 'run_depth': 5})
    (source / 'judgments.jsonl').write_text('{"j": 1}\n', encoding='utf-8')
    (source / 'judge_runs.jsonl').write_text('{"r": 1}\n', encoding='utf-8')
    return source, pool, runs


def test_derive_pool_keeps_rows_within_depth_and_trims_ranks(tmp_path):
    from legal_rag.evaluation.relevance_pool import derive_pool
    _, pool, runs = _source_pool(tmp_path)
    rows = {row['chunk_id']: row for row in derive_pool(pool, runs, pool_depth=2, include_unretrieved_gold=False)}
    assert set(rows) == {'a#0', 'gold#0', 'b#0', 'e#0'}
    assert rows['b#0']['ranks'] == {'sparse': 1}                  # dense rank 3 trimmed
    assert rows['gold#0']['ranks'] == {'dense': 2} and rows['gold#0']['is_gold_chunk']
    assert rows['a#0']['question'] == '질문'
    assert all(row['ranks'] for row in pool if row['chunk_id'] == 'b#0')   # source rows untouched
    assert [r['chunk_id'] for r in derive_pool(pool, runs, pool_depth=2, include_unretrieved_gold=False)] == [
        r['chunk_id'] for r in pool if r['chunk_id'] in rows]      # source order kept


def test_derive_pool_can_keep_unretrieved_gold_with_empty_ranks(tmp_path):
    from legal_rag.evaluation.relevance_pool import derive_pool
    _, pool, runs = _source_pool(tmp_path)
    rows = {row['chunk_id']: row for row in derive_pool(pool, runs, pool_depth=2, include_unretrieved_gold=True)}
    assert set(rows) == {'a#0', 'gold#0', 'b#0', 'e#0', 'gold#1', 'gold#2'}
    assert rows['gold#1']['ranks'] == {} and rows['gold#1']['is_gold_chunk']   # sparse rank 3 > depth
    assert rows['gold#2']['ranks'] == {} and rows['gold#2']['is_gold_chunk']


def test_derive_pool_refuses_ranks_missing_from_the_pool(tmp_path):
    from legal_rag.evaluation.relevance_pool import derive_pool
    _, pool, runs = _source_pool(tmp_path)
    with pytest.raises(ValueError, match='not in the source pool'):
        derive_pool(pool, runs, pool_depth=4, include_unretrieved_gold=False)
    with pytest.raises(ValueError):
        derive_pool(pool, runs, pool_depth=0, include_unretrieved_gold=False)


def test_derive_pool_dir_writes_provenance_and_copies_runs_and_judgments(tmp_path):
    import hashlib
    from legal_rag.evaluation.relevance_pool import derive_pool_dir
    source, _, runs = _source_pool(tmp_path)
    before = {p.name: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    out = derive_pool_dir(source, tmp_path / 'out', pool_depth=2, include_unretrieved_gold=False)
    meta = json.loads((out / 'pool.json').read_text(encoding='utf-8'))
    assert meta['target_sha256'] == 't' and meta['run_depth'] == 5
    assert meta['pool_depth'] == 2 and meta['pairs'] == 4 and meta['include_unretrieved_gold'] is False
    assert meta['modes'] == ['dense', 'sparse']
    assert meta['derived_from'] == {'pool_dir': str(source), 'pool_jsonl_sha256': hashlib.sha256(
        (source / 'pool.jsonl').read_bytes()).hexdigest(), 'pool_depth': 3}
    assert meta['judgments_copied_from'] == str(source)
    assert len(read_jsonl(out / 'pool.jsonl')) == 4
    assert (out / 'runs' / 'dense.jsonl').read_bytes() == (source / 'runs' / 'dense.jsonl').read_bytes()
    assert (out / 'judgments.jsonl').read_text(encoding='utf-8') == '{"j": 1}\n'
    assert (out / 'judge_runs.jsonl').read_text(encoding='utf-8') == '{"r": 1}\n'
    assert not (out / 'calibration.json').exists()
    assert {p.name: p.read_bytes() for p in source.rglob('*') if p.is_file()} == before
    with pytest.raises(FileExistsError):
        derive_pool_dir(source, out, pool_depth=2, include_unretrieved_gold=False)


def test_derive_pool_dir_refuses_to_deepen(tmp_path):
    from legal_rag.evaluation.relevance_pool import derive_pool_dir
    source, _, _ = _source_pool(tmp_path)
    with pytest.raises(ValueError, match='deepen'):
        derive_pool_dir(source, tmp_path / 'out', pool_depth=4, include_unretrieved_gold=False)
    assert not (tmp_path / 'out').exists()


def test_derive_pool_dir_without_judgments_records_none(tmp_path):
    from legal_rag.evaluation.relevance_pool import derive_pool_dir
    source, _, _ = _source_pool(tmp_path)
    (source / 'judgments.jsonl').unlink()
    (source / 'judge_runs.jsonl').unlink()
    out = derive_pool_dir(source, tmp_path / 'out', pool_depth=3, include_unretrieved_gold=True)
    assert json.loads((out / 'pool.json').read_text(encoding='utf-8'))['judgments_copied_from'] is None
    assert not (out / 'judgments.jsonl').exists()


def test_derive_script_writes_pool_and_refuses_existing_output(tmp_path):
    import importlib.util
    from pathlib import Path
    from typer.testing import CliRunner
    path = Path(__file__).resolve().parents[3] / 'scripts' / 'derive_relevance_pool.py'
    spec = importlib.util.spec_from_file_location('script_derive_relevance_pool', path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    source, _, _ = _source_pool(tmp_path)
    args = ['--from', str(source), '--output', str(tmp_path / 'out'), '--pool-depth', '2']
    result = CliRunner().invoke(script.app, args)
    assert result.exit_code == 0, result.output
    assert json.loads((tmp_path / 'out' / 'pool.json').read_text(encoding='utf-8'))['pairs'] == 4
    assert CliRunner().invoke(script.app, args).exit_code == 1
    assert CliRunner().invoke(script.app, ['--help']).exit_code == 0


def _adopt_fixture():
    from legal_rag.evaluation.relevance_pool import adopt_run
    pool, runs = build_pool(CASES, {'fanout': ListRetriever(['a#0', 'b#0'])}, gold_chunks={},
                            pool_depth=2, run_depth=2, case_filters=no_filters)
    source_pool = [dict(row, ranks={'old': 1}) for row in
                   build_pool(CASES, {'old': ListRetriever(['b#0', 'z#0'])}, gold_chunks={},
                              pool_depth=2, run_depth=2)[0]]
    copied = [{'case_id': 'c1', 'ranked': [{'chunk_id': 'z#0', 'document_id': 'z'},
                                           {'chunk_id': 'a#0', 'document_id': 'a'}]}]
    return adopt_run, pool, runs, source_pool, copied


def test_adopt_run_brings_in_the_pool_rows_the_copied_run_needs():
    adopt_run, pool, runs, source_pool, copied = _adopt_fixture()
    merged, runs = adopt_run(pool, runs, 'filtered', copied, source_pool, pool_depth=2)
    rows = {row['chunk_id']: row for row in merged}
    assert set(rows) == {'a#0', 'b#0', 'z#0'}
    assert rows['z#0']['ranks'] == {'filtered': 1} and rows['a#0']['ranks'] == {'fanout': 1, 'filtered': 2}
    assert rows['b#0']['ranks'] == {'fanout': 2}
    assert set(runs) == {'fanout', 'filtered'} and runs['filtered'] == copied
    assert [row['ranks'] for row in pool] == [{'fanout': 1}, {'fanout': 2}]      # inputs untouched


def test_adopt_run_refuses_a_pair_no_pool_can_describe():
    adopt_run, pool, runs, source_pool, copied = _adopt_fixture()
    with pytest.raises(ValueError, match='neither pool'):
        adopt_run(pool, runs, 'filtered', [{'case_id': 'c1', 'ranked': [
            {'chunk_id': 'unknown#0', 'document_id': 'unknown'}]}], source_pool, pool_depth=2)
    with pytest.raises(ValueError, match='already recorded'):
        adopt_run(pool, runs, 'fanout', copied, source_pool, pool_depth=2)

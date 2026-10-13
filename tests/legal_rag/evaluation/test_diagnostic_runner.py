import json
from pathlib import Path

import pytest

from legal_rag.config import Settings
from legal_rag.evaluation.diagnosis.diagnostic_runner import load_diagnostic_rows, select_review_cases, validate_index_metadata


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return path


def qa(doc, question='Q', **extra):
    return dict(document_id=doc, question=question, answer='never query', legal_domain='civil', **extra)


def test_explicit_eval_membership_full_duplicate_lookup(tmp_path):
    source = jsonl(tmp_path/'qa.jsonl', [qa('A'), qa('B'), qa('C', legal='x')])
    dataset = jsonl(tmp_path/'eval.jsonl', [dict(id='e', query=' Q ', relevant_document_ids=['B'])])
    rows, duplicates, provenance = load_diagnostic_rows(source, domain='civil', dataset=dataset)
    assert [row['document_id'] for row in rows] == ['B']
    assert duplicates['Q']['count'] == 3
    assert duplicates['Q']['document_ids'] == ['A', 'B', 'C']
    assert provenance['selection'] == 'evaluation_dataset'


def test_train_rows_are_not_silently_selected(tmp_path):
    source = jsonl(tmp_path/'qa.jsonl', [qa('A', split='train'), qa('B', split='test'), qa('C')])
    rows, _, provenance = load_diagnostic_rows(source, domain='civil')
    assert [row['document_id'] for row in rows] == ['B']
    assert provenance['selection'] == 'qa_evaluation_split'
    source = jsonl(source, [qa('A', split='train')])
    with pytest.raises(ValueError, match='evaluation'):
        load_diagnostic_rows(source, domain='civil')


def test_unmatched_eval_row_fails_instead_of_silent_fallback(tmp_path):
    source = jsonl(tmp_path/'qa.jsonl', [qa('A')])
    dataset = jsonl(tmp_path/'eval.jsonl', [dict(query='other', relevant_document_ids=['A'])])
    with pytest.raises(ValueError, match='match'):
        load_diagnostic_rows(source, domain='civil', dataset=dataset)


def test_failure_sampling_is_after_evaluation_and_reproducible():
    rows = [dict(case_id=str(i), status='ok', gold_found_in_top_k=i % 2 == 0) for i in range(30)]
    selected = select_review_cases(rows, limit=5, seed=42, only_failures=True)
    assert len(selected) == 5
    assert all(not row['gold_found_in_top_k'] for row in selected)
    assert selected == select_review_cases(list(reversed(rows)), limit=5, seed=42, only_failures=True)


def test_index_version_collection_and_dimension_checked():
    settings = Settings(corpus={'version':'v1'})
    metadata = dict(corpus_version='v1', collection='legal_v1_fake', dimension=64)
    validate_index_metadata(settings, metadata, collection='legal_v1_fake', dimension=64)
    for changed in ({'corpus_version':'v2'}, {'collection':'wrong'}, {'dimension':3}):
        with pytest.raises(ValueError, match='index'):
            validate_index_metadata(settings, {**metadata, **changed}, collection='legal_v1_fake', dimension=64)


def test_live_service_and_cli_with_embedded_fixture(tmp_path):
    import importlib.util
    from typer.testing import CliRunner
    from legal_rag.factory import build_embedder
    from legal_rag.retrieval.index_builder import build_index
    from legal_rag.schemas import Chunk, ChunkMetadata
    from legal_rag.evaluation.diagnosis.diagnostic_runner import run_retrieval_diagnostics
    from legal_rag.cli import settings_from_options

    settings = settings_from_options(profile='default', root=tmp_path, corpus_version='v1',
                                     extra={'embedding': {'backend': 'fake'},
                                            'retrieval': {'k_dense': 1, 'k_sparse': 1, 'k_fused': 1}})
    corpus = settings.processed_corpus_dir()
    corpus.mkdir(parents=True)
    (corpus/'documents.jsonl').write_text('{}\n')
    chunk = Chunk(id='A#0', document_id='A', document_type='statute', path='0', order=0,
                  text='civil question evidence', title='Act',
                  metadata=ChunkMetadata(dataset='test', corpus_version='v1', legal_domain='civil'))
    (corpus/'chunks.jsonl').write_text(''.join(
        chunk.model_copy(update={'id': f'A#{i}', 'order': i}).model_dump_json()+'\n'
        for i in range(4)))
    jsonl(corpus/'qa.jsonl', [qa('A', 'civil question'), qa('missing', 'absent')])
    build_index(settings, embedder=build_embedder(settings))
    for mode in ('dense', 'sparse', 'hybrid'):
        output = run_retrieval_diagnostics(settings, domain='civil', retrieval_mode=mode,
                                          top_k=1, deep_k=3, output=tmp_path/mode)
        summary = json.loads((output/'summary.json').read_text())
        assert summary['evaluated'] == 1
        assert summary['missing_gold_corpus'] == 1
        assert summary['exact_recall_at_k'] == 1
        assert len((output/'missing_gold_corpus.jsonl').read_text().splitlines()) == 1
        assert summary['candidate_depths']['per_leg'] == 3
        assert summary['candidate_depths']['fused'] == 3
        record = json.loads((output/'failures.jsonl').read_text())
        assert len(record['retrieved']) == 3
        assert 'rerank_score' not in record['retrieved'][0]
    script = Path(__file__).resolve().parents[3] / 'scripts' / 'diagnose_retrieval.py'
    spec = importlib.util.spec_from_file_location('diagnose_cli', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runner = CliRunner()
    result = runner.invoke(module.app, ['--help'])
    assert result.exit_code == 0
    assert '--deep-k' in result.stdout
    result = runner.invoke(module.app, ['--domain', 'civil', '--profile', 'default',
        '--root', str(tmp_path), '--corpus-version', 'v1', '--output', str(tmp_path/'cli')],
        env={'LEGAL_RAG__EMBEDDING__BACKEND': 'fake'})
    assert result.exit_code == 0, result.output
    assert (tmp_path/'cli/review.csv').exists()
    result = runner.invoke(module.app, ['--domain', 'civil', '--top-k', '20', '--deep-k', '5'])
    assert result.exit_code != 0


def test_dataset_membership_does_not_override_qa_splits(tmp_path):
    source = jsonl(tmp_path/'qa.jsonl', [qa('A', split='train'), qa('A', split='test')])
    dataset = jsonl(tmp_path/'eval.jsonl', [dict(query='Q', relevant_document_ids=['A'])])
    rows, duplicates, _ = load_diagnostic_rows(source, domain='civil', dataset=dataset)
    assert len(rows) == 1
    assert rows[0]['split'] == 'test'
    assert duplicates['Q']['count'] == 2


def test_mixed_domain_dataset_is_filtered(tmp_path):
    source = jsonl(tmp_path/'qa.jsonl', [qa('A'), dict(qa('B'), legal_domain='criminal')])
    dataset = jsonl(tmp_path/'eval.jsonl', [dict(query='Q', relevant_document_ids=['A', 'B'])])
    rows, _, _ = load_diagnostic_rows(source, domain='civil', dataset=dataset)
    assert [row['document_id'] for row in rows] == ['A']


def _mini_corpus(directory, chunks):
    from legal_rag.ingestion.corpus import Corpus
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'documents.jsonl').write_text('', encoding='utf-8')
    (directory / 'chunks.jsonl').write_text(
        ''.join(chunk.model_dump_json() + '\n' for chunk in chunks), encoding='utf-8')
    return Corpus.load(directory)


def _mini_chunk(chunk_id, document_id, text='본문'):
    from legal_rag.schemas import Chunk, ChunkMetadata, DocumentType
    return Chunk(id=chunk_id, document_id=document_id, document_type=DocumentType.statute, path='p',
                 order=0, text=text, title='제목',
                 metadata=ChunkMetadata(dataset='d', corpus_version='v'))


def test_gold_chunks_are_found_in_whichever_corpus_holds_the_document(tmp_path):
    from legal_rag.evaluation.diagnosis.diagnostic_runner import gold_chunks_across_corpora
    corpora = {'criminal-v1': _mini_corpus(tmp_path / 'c', [_mini_chunk('c#0', 'crime', '형사 본문'),
                                                            _mini_chunk('c#1', 'other')]),
               'civil-v1': _mini_corpus(tmp_path / 'v', [_mini_chunk('v#0', 'civil', '민사 본문')])}
    found = gold_chunks_across_corpora(corpora, {'crime', 'civil'}, max_gold_chunks=10, preview_chars=10 ** 9)
    assert sorted(found) == ['civil', 'crime']
    assert [chunk.text for chunk in found['crime']] == ['형사 본문']
    assert [chunk.text for chunk in found['civil']] == ['민사 본문']


def test_a_gold_document_in_two_corpora_is_refused(tmp_path):
    from legal_rag.evaluation.diagnosis.diagnostic_runner import gold_chunks_across_corpora
    corpora = {'a-v1': _mini_corpus(tmp_path / 'a', [_mini_chunk('a#0', 'shared')]),
               'b-v1': _mini_corpus(tmp_path / 'b', [_mini_chunk('b#0', 'shared')])}
    with pytest.raises(ValueError, match='ambiguous'):
        gold_chunks_across_corpora(corpora, {'shared'}, max_gold_chunks=10, preview_chars=100)

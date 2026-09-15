import json
from pathlib import Path

import pytest

from legal_rag.config import load_settings
from legal_rag.ingestion.corpus import Corpus
from legal_rag.ingestion import import_processed as importer
from legal_rag.factory import build_hybrid_retriever
from legal_rag.retrieval.index_builder import build_index
from legal_rag.schemas import RetrievalFilters


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    docs = [dict(document_id=f'doc-{i}', document_type=kind, legal_domain='intellectual_property',
                 title='특허 판결', authority='특허심판원', date='2024-08-07 00:00:00',
                 text=f'특허권 침해 원문 {i}', metadata={'nested': {'tags': ['원문']}})
            for i, kind in enumerate(['statute', 'judgment', 'legal_interpretation', 'decision', 'patent_trial_decision', 'constitutional_decision'])]
    chunks = [dict(d, chunk_id=f"{d['document_id']}_chunk_0000", chunk_index=0, total_chunks=1) for d in docs]
    write_rows(root / 'documents.jsonl', docs)
    write_rows(root / 'chunks.jsonl', chunks)
    (root / 'qa.jsonl').write_text('THIS MUST NEVER BE READ')
    return root, docs, chunks


def settings(tmp_path, version='imported'):
    return load_settings(profile='default', overrides={'paths': {'root': str(tmp_path)},
                         'corpus': {'version': version}, 'embedding': {'backend': 'fake'}})


def test_import_preserves_source_and_supports_retrieval(source, tmp_path):
    root, docs, chunks = source
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    cfg = settings(tmp_path)
    report = importer.import_processed(root, cfg)
    assert report['documents'] == 6 and report['chunks'] == 6
    corpus = Corpus.load(cfg.processed_corpus_dir())
    imported = list(corpus.chunks())
    assert [c.id for c in imported] == [c['chunk_id'] for c in chunks]
    assert [c.document_type.value for c in imported] == [c['document_type'] for c in chunks]
    assert imported[0].metadata.original == {k: v for k, v in chunks[0].items() if k != 'text'}
    assert corpus.document('doc-0').text == docs[0]['text']
    assert corpus.document('doc-0').metadata['original']['metadata'] == docs[0]['metadata']
    assert corpus.manifest().counts.chunks == 6
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before
    build_index(cfg)
    retriever = build_hybrid_retriever(cfg)
    try:
        hits = retriever.retrieve('특허권 침해', k=3, filters=RetrievalFilters(
            legal_domain='intellectual_property', document_types=['patent_trial_decision']))
        assert hits and hits[0].chunk.id == 'doc-4_chunk_0000'
        assert hits[0].chunk.metadata.original == imported[4].metadata.original
    finally:
        retriever.legs[0].store.close()


@pytest.mark.parametrize('defect', ['duplicate_document', 'duplicate_chunk', 'empty', 'orphan', 'domain', 'type', 'count', 'position', 'malformed'])
def test_invalid_input_never_publishes(source, tmp_path, defect):
    root, docs, chunks = source
    if defect == 'duplicate_document': docs.append(docs[0])
    if defect == 'duplicate_chunk': chunks.append(chunks[0])
    if defect == 'empty': chunks[0]['text'] = '  '
    if defect == 'orphan': chunks[0]['document_id'] = 'missing'
    if defect == 'domain': chunks[0]['legal_domain'] = 'criminal'
    if defect == 'type': chunks[0]['document_type'] = 'judgment'
    if defect == 'count': chunks[0]['total_chunks'] = 2
    if defect == 'position': chunks[0]['chunk_index'] = -1
    write_rows(root / 'documents.jsonl', docs)
    write_rows(root / 'chunks.jsonl', chunks)
    if defect == 'malformed': (root / 'chunks.jsonl').write_text('{bad json}\n')
    cfg = settings(tmp_path)
    with pytest.raises(ValueError): importer.import_processed(root, cfg)
    assert not cfg.processed_corpus_dir().exists()


def test_sample_complete_documents_and_no_overwrite(source, tmp_path):
    root, _, _ = source
    cfg = settings(tmp_path)
    report = importer.import_processed(root, cfg, max_documents=2)
    assert report['documents'] == 2 and report['chunks'] == 2
    assert report['source_documents'] == 6 and report['source_chunks'] == 6
    saved = (cfg.processed_corpus_dir() / 'chunks.jsonl').read_bytes()
    with pytest.raises(FileExistsError): importer.import_processed(root, cfg)
    assert (cfg.processed_corpus_dir() / 'chunks.jsonl').read_bytes() == saved


def test_validation_only_writes_no_corpus(source, tmp_path):
    root, _, _ = source
    cfg = settings(tmp_path)
    report = importer.import_processed(root, cfg, validate_only=True)
    assert report['source_chunks'] == 6
    assert not cfg.processed_corpus_dir().exists()


def test_documents_without_chunks_are_preserved_and_reported(source, tmp_path):
    root, docs, chunks = source
    write_rows(root / 'chunks.jsonl', chunks[1:])
    cfg = settings(tmp_path)
    report = importer.import_processed(root, cfg)
    assert report['documents_without_chunks'] == ['doc-0']
    assert report['documents'] == 6 and report['chunks'] == 5
    assert Corpus.load(cfg.processed_corpus_dir()).document('doc-0').text == docs[0]['text']


def test_sample_with_no_chunks_rejected(source, tmp_path):
    root, _, chunks = source
    write_rows(root / 'chunks.jsonl', chunks[1:])
    with pytest.raises(ValueError, match='selected documents have no chunks'):
        importer.import_processed(root, settings(tmp_path), max_documents=1)


def test_generic_date_does_not_invent_legal_effective_date(source, tmp_path):
    root, _, _ = source
    cfg = settings(tmp_path)
    importer.import_processed(root, cfg)
    chunk = next(Corpus.load(cfg.processed_corpus_dir()).chunks())
    assert chunk.metadata.effective_date is None
    assert chunk.metadata.original['date'] == '2024-08-07 00:00:00'


@pytest.mark.parametrize('line', [
    '{"document_id":"first","document_id":"second"}',
    '{"metadata":{"key":1,"key":2}}',
])
def test_duplicate_json_keys_rejected_without_publication(source, tmp_path, line):
    root, _, _ = source
    (root / 'documents.jsonl').write_text(line + '\n')
    cfg = settings(tmp_path)
    with pytest.raises(ValueError, match='duplicate JSON key'):
        importer.import_processed(root, cfg)
    assert not cfg.processed_corpus_dir().exists()


def test_search_cli_accepts_preserved_external_ids(source, tmp_path):
    import subprocess
    import sys

    root, _, _ = source
    cfg = settings(tmp_path)
    importer.import_processed(root, cfg)
    build_index(cfg)
    script = Path(__file__).resolve().parents[2] / 'scripts/search.py'
    result = subprocess.run([sys.executable, str(script), '특허권', '--root', str(tmp_path),
                             '--profile', 'default', '--corpus-version', 'imported',
                             '--embedding-backend', 'fake'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '_chunk_0000' in result.stdout


def test_unknown_document_type_fails_with_file_and_line(source, tmp_path):
    # Administrative sources spell this "judgement". Import must reject it loudly
    # rather than tempt anyone into aliasing the typo inside the importer.
    root, _, chunks = source
    write_rows(root / 'chunks.jsonl', [dict(chunks[0], document_type='judgement')])
    with pytest.raises(ValueError) as excinfo:
        importer.import_processed(root, settings(tmp_path))
    assert 'judgement' in str(excinfo.value)
    assert 'chunks.jsonl:1' in str(excinfo.value)


def test_case_like_types_carry_authority_into_court(source, tmp_path):
    # Whatever renders as Case evidence needs a court, or the header loses it.
    root, _, _ = source
    importer.import_processed(root, settings(tmp_path, version='courts'))
    by_type = {c.document_type.value: c for c in Corpus.load(
        settings(tmp_path, version='courts').processed_corpus_dir()).chunks()}
    for kind in ('judgment', 'decision', 'patent_trial_decision', 'constitutional_decision'):
        assert by_type[kind].metadata.court == '특허심판원', kind
    assert by_type['statute'].metadata.court is None
    assert by_type['legal_interpretation'].metadata.court is None

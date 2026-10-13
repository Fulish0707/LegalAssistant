import weakref

import pytest

from legal_rag.embeddings import FakeEmbedder
from legal_rag.factory import open_store
from legal_rag.ingestion.corpus import Corpus
from legal_rag.retrieval.index_builder import build_index
from legal_rag.retrieval.sparse import BM25Stats
from legal_rag.schemas import Chunk, ChunkMetadata
from legal_rag.config import load_settings


def test_bm25_stream_matches_existing_statistics():
    docs = [['a', 'a', 'b'], ['b', 'c'], []]
    expected = BM25Stats.fit(dict(enumerate(docs)), k1=1.2, b=0.75)
    actual = BM25Stats.fit_stream(iter(docs), k1=1.2, b=0.75)
    assert actual.df == {"a": 1, "b": 2, "c": 1}
    assert actual.n_docs == 3
    assert actual.avg_len == 5 / 3
    assert actual.__dict__ == expected.__dict__
    assert actual.document_vector(docs[0]) == expected.document_vector(docs[0])


def configuration(tmp_path):
    return load_settings(profile='default', overrides={'paths': {'root': str(tmp_path)},
                         'embedding': {'backend': 'fake', 'batch_size': 4}})


def streamed_corpus(monkeypatch):
    refs = []

    class Stream:
        def chunks(self):
            for index in range(200):
                assert sum(ref() is not None for ref in refs) <= 33, 'indexer retains too many Chunk objects'
                chunk = Chunk(id=f'doc#{index}', document_id='doc', document_type='statute',
                              path=f'chunk:{index}', order=index, text='특허권 침해', title='특허',
                              metadata=ChunkMetadata(dataset='test', corpus_version='v1'))
                refs.append(weakref.ref(chunk))
                yield chunk

    monkeypatch.setattr(Corpus, 'load', lambda path: Stream())


def test_indexer_does_not_materialize_entire_corpus(monkeypatch, tmp_path):
    streamed_corpus(monkeypatch)
    cfg = configuration(tmp_path)
    report = build_index(cfg)
    assert report.chunks == 200
    store = open_store(cfg, FakeEmbedder())
    try:
        assert store.count() == 200
    finally:
        store.close()


def test_embedding_failure_releases_embedded_store(monkeypatch, tmp_path):
    streamed_corpus(monkeypatch)
    cfg = configuration(tmp_path)

    class FailingEmbedder(FakeEmbedder):
        def embed_documents(self, texts):
            raise RuntimeError('embedding unavailable')

    with pytest.raises(RuntimeError, match='embedding unavailable'):
        build_index(cfg, embedder=FailingEmbedder())
    store = open_store(cfg, FakeEmbedder())
    store.close()

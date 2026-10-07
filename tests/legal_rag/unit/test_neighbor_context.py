"""Focused checks for the approved structural context expansion."""
import pytest
from legal_rag.rag.context import ContextBuilder
from legal_rag.schemas import ScoredChunk
from legal_rag.retrieval.store import VectorStore
from legal_rag.retrieval.sparse import BM25Stats
from legal_rag.retrieval.tokenizer import RegexTokenizer
from legal_rag.embeddings import FakeEmbedder
from tests.legal_rag.unit.test_store import _chunks


def chunk(order, doc='case', version='v'):
    base = _chunks()[2]
    return base.model_copy(update={'id': f'{version}:{doc}#{order}', 'document_id': doc,
        'order': order, 'path': str(order), 'text': f'판단 {order}\n' + '관련 법적 조건과 예외. ' * 8,
        'metadata': base.metadata.model_copy(update={'corpus_version': version})})


def hit(c):
    return ScoredChunk(chunk=c, score=.8, retriever='reranked')


def test_expansion_preserves_anchors_and_deduplicates_neighbors():
    chunks = [chunk(i) for i in range(4)]
    def lookup(c, radius):
        return [other for other in chunks if 0 < abs(other.order-c.order) <= radius]
    bundle = ContextBuilder(neighbor_radius=1, neighbor_lookup=lookup).build([hit(chunks[1]), hit(chunks[2])])
    assert [e.chunk_id for e in bundle.items] == [chunks[i].id for i in [1, 2, 0, 3]]
    assert [e.source_id for e in bundle.items] == ['S1','S2','S3','S4']
    assert bundle.items[2].score == bundle.items[3].score == 0
    assert bundle.items[2].header['Case number'] == chunks[0].metadata.case_number
    assert bundle.items[2].header['Context'] == 'Adjacent to S1'
    assert len(bundle.render()) <= 7500


def test_budget_prioritizes_anchors_over_neighbors():
    anchor, neighbor = chunk(1), chunk(0)
    budget = len(ContextBuilder().build([hit(anchor)]).render())
    bundle = ContextBuilder(neighbor_radius=1, max_context_chars=budget,
        neighbor_lookup=lambda c,r:[neighbor]).build([hit(anchor)])
    assert [e.chunk_id for e in bundle.items] == [anchor.id]
    assert len(bundle.render()) == budget


def test_budget_also_applies_without_expansion():
    a,b = chunk(0),chunk(1)
    budget = len(ContextBuilder().build([hit(a)]).render())
    bundle = ContextBuilder(max_context_chars=budget).build([hit(a),hit(b)])
    assert [e.chunk_id for e in bundle.items] == [a.id]


def test_allowlist_avoids_lookup_for_other_corpora():
    def lookup(c,r):
        raise AssertionError('Unexpected lookup')
    bundle = ContextBuilder(neighbor_radius=1, neighbor_corpora=['new'],
        neighbor_lookup=lookup).build([hit(chunk(1))])
    assert len(bundle.items) == 1


@pytest.mark.parametrize('wrong', [chunk(0,doc='other'),chunk(0,version='other'),chunk(4)])
def test_invalid_neighbor_window_is_rejected(wrong):
    with pytest.raises(ValueError,match='outside the requested document window'):
        ContextBuilder(neighbor_radius=1,neighbor_lookup=lambda c,r:[wrong]).build([hit(chunk(1))])


def test_store_neighbors_are_document_and_version_scoped(tmp_path):
    chunks = [chunk(i) for i in range(4)] + [chunk(0,doc='other'),chunk(0,version='other')]
    embedder,tok = FakeEmbedder(8),RegexTokenizer()
    stats = BM25Stats.fit({c.id:tok.tokenize(c.text) for c in chunks},k1=1.2,b=.75)
    store = VectorStore.embedded(tmp_path/'qdrant','neighbor-test',8)
    try:
        store.recreate()
        store.upsert(chunks,dense=embedder.embed_documents([c.text for c in chunks]),
            sparse=[stats.document_vector(tok.tokenize(c.text)) for c in chunks])
        assert store.chunks_near(chunks[1],1) == [chunks[0],chunks[2]]
    finally:
        store.close()

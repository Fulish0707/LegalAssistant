from legal_rag.embeddings import FakeEmbedder
from legal_rag.retrieval.document import find_document, full_text, pinned_chunks
from legal_rag.retrieval.sparse import BM25Stats
from legal_rag.retrieval.store import VectorStore
from legal_rag.retrieval.tokenizer import RegexTokenizer
from legal_rag.schemas import Chunk, ChunkMetadata, DocumentType


def piece(order, text, doc="d", version="v"):
    return Chunk(id=f"{version}:{doc}#{order}", document_id=doc, document_type=DocumentType.judgment, path=str(order),
                 order=order, text=text, title="t", metadata=ChunkMetadata(dataset="f", corpus_version=version))


def test_overlapping_windows_are_joined_without_repeating_the_overlap():
    whole = "가" * 30 + "나" * 30 + "다" * 30
    chunks = [piece(0, whole[:50]), piece(1, whole[30:80]), piece(2, whole[60:])]

    text, start, end = full_text(chunks, chunks[1].id)

    assert text == whole
    assert text[start:end] == chunks[1].text


def test_chunks_without_overlap_are_set_apart():
    chunks = [piece(0, "제1조(목적) 이 법은 목적을 정한다."), piece(1, "제2조(정의) 이 법에서 쓰는 말의 뜻을 정한다.")]

    text, start, end = full_text(chunks, chunks[1].id)

    assert text == chunks[0].text + "\n\n" + chunks[1].text
    assert text[start:end] == chunks[1].text


def test_a_short_shared_run_is_not_treated_as_overlap():
    chunks = [piece(0, "임대차 계약은 종료한다"), piece(1, "종료한다고 통지하였다")]

    assert full_text(chunks, chunks[0].id)[0] == chunks[0].text + "\n\n" + chunks[1].text


def test_the_document_is_read_from_the_store_in_order_and_only_from_its_own_corpus(tmp_path):
    whole = "".join(chr(0xAC00 + i) for i in range(1000))
    # 300 chunks of one document: more than one page of a store scroll.
    document = [piece(i, whole[i * 3:i * 3 + 20]) for i in range(300)]
    others = [piece(0, "다른 문서", doc="other"), piece(0, "다른 코퍼스", version="w")]
    chunks = document[::-1] + others
    embedder, tok = FakeEmbedder(8), RegexTokenizer()
    stats = BM25Stats.fit({c.id: tok.tokenize(c.text) for c in chunks}, k1=1.2, b=.75)
    store = VectorStore.embedded(tmp_path / "qdrant", "document-test", 8)
    try:
        store.recreate()
        store.upsert(chunks, dense=embedder.embed_documents([c.text for c in chunks]),
                     sparse=[stats.document_vector(tok.tokenize(c.text)) for c in chunks])

        assert store.document_chunks(document[5]) == document
        chunk, text, start, end = find_document([store], document[7].id)
        assert chunk == document[7] and text[start:end] == document[7].text
        assert text == whole[:299 * 3 + 20]
        assert find_document([store], "no-such-chunk") is None
    finally:
        store.close()


class OneChunkStore:
    def __init__(self, *chunks):
        self.held = {chunk.id: chunk for chunk in chunks}

    def chunk(self, chunk_id):
        return self.held.get(chunk_id)


def test_a_pinned_case_named_only_in_its_document_id_gets_its_number():
    civil = piece(0, "판단", doc="서울중앙지방법원-2023가단5103393")
    numbered = piece(0, "판단", doc="criminal::judgment::1").model_copy(update={"metadata": ChunkMetadata(
        dataset="f", corpus_version="v", case_number="84도2229")})
    plain = piece(0, "제1조", doc="statute:abc")

    pinned = pinned_chunks([OneChunkStore(plain), OneChunkStore(civil, numbered)], [civil.id, "no-such-chunk", numbered.id, plain.id])

    assert [chunk.id for chunk in pinned] == [civil.id, numbered.id, plain.id]
    assert [chunk.metadata.case_number for chunk in pinned] == ["2023가단5103393", "84도2229", None]
    assert pinned[0].text == civil.text

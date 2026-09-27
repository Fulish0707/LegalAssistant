"""Display labels for hits whose chunk carries no court, case number, law name or title.

The criminal and administrative imports keep that information only in the header of a
document's first chunk, so a middle chunk has to borrow it. The header texts below are
copied from the imported corpora (2026-09-27).
"""
from __future__ import annotations

from legal_rag.retrieval.labels import document_labels, fallback_label, first_chunk_id, parse_header
from legal_rag.retrieval.sparse import SparseVec
from legal_rag.retrieval.store import VectorStore
from legal_rag.schemas import Chunk, ChunkMetadata, DocumentType, ScoredChunk


def test_first_chunk_id_resets_the_trailing_chunk_index():
    assert first_chunk_id("criminal::HS_P_323457::chunk_0004") == "criminal::HS_P_323457::chunk_0000"
    assert first_chunk_id("13-진정-0886700_chunk_0001") == "13-진정-0886700_chunk_0000"
    assert first_chunk_id("d1#art:제750조") is None


def test_header_with_court_case_number_and_name():
    text = "광주지방법원\n판결\n사건 2022고단4222 스토킹범죄의 처벌 등에 관한 법률 위반\n피고인 A"
    assert parse_header(text) == "광주지방법원 2022고단4222 스토킹범죄의 처벌 등에 관한 법률 위반"


def test_letter_spaced_court_and_label_on_its_own_line():
    text = "서 울 북 부 지 방 법 원\n판 결\n사 건\n2022고단2471 사기, 전자금융거래법위반\n2022초기1348 배상명령신청"
    assert parse_header(text) == "서울북부지방법원 2022고단2471 사기, 전자금융거래법위반"


def test_court_division_line_is_skipped():
    text = "수원고등법원\n제3행정부\n판결\n사건 2023누12770 양도소득세 경정거부처분 취소\n제1심판결 수원지방법원 2023.5.18.\n선고 2022구합75274 판결"
    assert parse_header(text) == "수원고등법원 2023누12770 양도소득세 경정거부처분 취소"


def test_constitutional_number_implies_the_constitutional_court():
    text = "사  건○  2020헌마843 가석방 심사대상 제외 위헌확인 등\n청 구 인  김○○○"
    assert parse_header(text) == "헌법재판소 2020헌마843 가석방 심사대상 제외 위헌확인 등"


def test_a_case_number_not_introduced_by_the_case_label_is_never_used():
    """First chunks also cite other decisions; naming the wrong case is worse than none."""
    text = "【피 고 인】 피고인 1 외 3인\n【원심판결】 서울고법 2019노1234 판결\n【주 문】"
    assert parse_header(text) is None
    assert parse_header("이 사건 심판청구를 모두 각하한다. 대법원 2018다12345 참조") is None


def test_fallback_names_domain_and_kind_not_a_text_fragment():
    assert fallback_label("statute", "administrative") == "행정 법령"
    assert fallback_label("judgment", "criminal") == "형사 판결"
    assert fallback_label("constitutional_decision", "criminal") == "헌법재판소 결정"
    assert fallback_label("decision", None) == "결정"


def _chunk(chunk_id, order, text, doc_type=DocumentType.judgment, **meta):
    return Chunk(id=chunk_id, document_id="criminal::judgment::1", document_type=doc_type, path=f"chunk:{order:04d}",
                 order=order, text=text, title="",
                 metadata=ChunkMetadata(dataset="f", corpus_version="criminal-v1", legal_domain="criminal", **meta))


def _store(tmp_path, chunks):
    store = VectorStore.embedded(tmp_path / "q", collection="c", dimension=4)
    store.recreate()
    store.upsert(chunks, dense=[[1.0, 0.0, 0.0, 0.0]] * len(chunks), sparse=[SparseVec(indices=[1], values=[1.0])] * len(chunks))
    return store


def test_middle_chunk_borrows_its_documents_header(tmp_path):
    first = _chunk("criminal::HS_P_1::chunk_0000", 0, "광주지방법원\n판결\n사건 2022고단4222 사기\n피고인 A")
    middle = _chunk("criminal::HS_P_1::chunk_0003", 3, "피고인은 피해자를 기망하여")
    store = _store(tmp_path, [first, middle])

    labels = document_labels([ScoredChunk(chunk=middle, score=0.9, retriever="reranked")], {"criminal-v1": store})

    assert labels == {"criminal::judgment::1": "광주지방법원 2022고단4222 사기"}


def test_unparseable_header_falls_back_and_labelled_hits_are_left_alone(tmp_path):
    first = _chunk("criminal::HS_P_1::chunk_0000", 0, "【피 고 인】 피고인 1")
    labelled = _chunk("x_chunk_0000", 0, "본문", court="대법원", case_number="2018다12345")
    labelled = labelled.model_copy(update={"document_id": "other"})
    store = _store(tmp_path, [first])

    labels = document_labels(
        [ScoredChunk(chunk=first, score=0.9, retriever="reranked"), ScoredChunk(chunk=labelled, score=0.8, retriever="reranked")],
        {"criminal-v1": store},
    )

    assert labels == {"criminal::judgment::1": "형사 판결"}

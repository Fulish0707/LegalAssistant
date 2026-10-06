"""A turn that carries the user's own case documents.

The documents are facts of the case, not legal authority: the model reads them in their own
block, the verifier treats what they say as the user's words, and a turn without documents
is prompted exactly as before.
"""

from legal_rag.llm import FakeChatModel
from legal_rag.schemas import Citation, CitationKind, QueryAnalysis, UserDocument
from tests.legal_rag.integration.test_api import _client
from tests.legal_rag.integration.test_chat_turn import _answer, _orchestrator, retriever  # noqa: F401
from tests.legal_rag.integration.test_ingest_to_search import indexed  # noqa: F401

QUESTION = "임차권등기명령 신청서에 적어야 하는 사항은?"
LEASE = UserDocument(name="임대차계약서.pdf", text="제5조 임대인은 민법 제618조에 따라 계약 종료일에 보증금 1억 원을 반환한다.")


def _model(*answers):
    return FakeChatModel({"QueryAnalysis": [QueryAnalysis(domain="민사", issues=["보증금 반환"])], "LegalAnswer": list(answers)})


def _generation_messages(model):
    return [c for c in model.calls if c.response_schema.__name__ == "LegalAnswer"][0].messages


def test_a_turn_without_documents_is_prompted_as_before(retriever):
    settings, r = retriever
    plain, empty = _model(_answer()), _model(_answer())
    _orchestrator(settings, r, plain).handle(QUESTION, conversation_id="a")
    _orchestrator(settings, r, empty).handle(QUESTION, conversation_id="b", user_documents=[])
    assert _generation_messages(plain) == _generation_messages(empty)
    assert not any("USER DOCUMENT" in m.content for m in _generation_messages(plain))


def test_documents_reach_the_model_in_their_own_block(retriever):
    settings, r = retriever
    model = _model(_answer())
    _orchestrator(settings, r, model).handle(QUESTION, conversation_id="c", user_documents=[LEASE])
    blocks = [m.content for m in _generation_messages(model) if "[USER DOCUMENT D1]" in m.content]
    assert len(blocks) == 1 and "임대차계약서.pdf" in blocks[0] and "보증금 1억 원" in blocks[0]
    assert "법적 근거가 아닙니다" in blocks[0] and "[LEGAL SOURCE S1]" not in blocks[0]


def test_an_article_written_in_the_document_is_not_a_fabricated_citation(retriever):
    settings, r = retriever
    repeats = _answer(analysis="계약서 제5조는 민법 제618조를 들어 종료일에 보증금을 반환한다고 정합니다(D1). "
                               "주택임대차보호법 제3조의3에 따라 임차권등기명령을 신청할 수 있습니다.")
    with_document = _orchestrator(settings, r, _model(repeats)).handle(QUESTION, conversation_id="d", user_documents=[LEASE])
    assert with_document.kind == "answer" and with_document.trace.policy_action == "accept"
    without = _orchestrator(settings, r, _model(repeats, repeats)).handle(QUESTION, conversation_id="e")
    assert without.trace.policy_action != "accept"


def test_a_contracts_own_clause_number_does_not_excuse_an_invented_article(retriever):
    """A lease numbers its clauses 제1조, 제2조 ...; that must not let any law's 제4조 through."""
    settings, r = retriever
    lease = UserDocument(name="계약서.pdf", text="제4조 (계약의 갱신) 갱신하지 않으려면 2개월 전까지 알려야 한다.")
    invented = _answer(analysis="주택임대차보호법 제4조에 따라 갱신을 거절할 수 있습니다. "
                                "주택임대차보호법 제3조의3에 따라 임차권등기명령을 신청할 수 있습니다.")
    response = _orchestrator(settings, r, _model(invented, invented)).handle(QUESTION, conversation_id="g", user_documents=[lease])
    assert response.trace.policy_action != "accept"
    assert "주택임대차보호법 제4조" in response.trace.verification.checked_statutes
    assert "제4조에 따라 갱신" not in (response.answer.analysis if response.answer else "")


def test_a_document_listed_as_a_citation_is_dropped_not_punished(retriever):
    settings, r = retriever
    base = _answer()
    cites_document = base.model_copy(update={"citations": [*base.citations, Citation(source_id="D1", kind=CitationKind.case, reference="임대차계약서 제5조")]})
    response = _orchestrator(settings, r, _model(cites_document)).handle(QUESTION, conversation_id="f", user_documents=[LEASE])
    assert response.trace.policy_action == "accept" and [c.source_id for c in response.answer.citations] == ["S1"]


def test_the_api_takes_up_to_three_documents(indexed):
    document = {"name": "계약서.pdf", "text": "제5조 보증금 반환"}
    client = _client(indexed, _model(_answer()))
    assert client.post("/v1/chat", json={"message": QUESTION, "user_documents": [document] * 4}).status_code == 422
    assert client.post("/v1/chat", json={"message": QUESTION, "user_documents": [{"name": "x"}]}).status_code == 422
    model = _model(_answer())
    body = _client(indexed, model).post("/v1/chat", json={"message": QUESTION, "user_documents": [document]}).json()
    assert body["kind"] == "answer" and any("[USER DOCUMENT D1]" in m.content for m in _generation_messages(model))

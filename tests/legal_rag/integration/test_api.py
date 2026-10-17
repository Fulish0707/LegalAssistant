import json

from fastapi.testclient import TestClient

from legal_rag.api import create_app
from legal_rag.config import load_settings
from legal_rag.llm import FakeChatModel
from legal_rag.rag.conversation import InMemoryConversationStore
from legal_rag.rag.orchestrator import Orchestrator
from legal_rag.schemas import Confidence, LegalAnswer, QueryAnalysis
from legal_rag.verification import VerificationPolicy, Verifier
from tests.legal_rag.integration.test_chat_turn import _answer


def _client(indexed, model):
    settings, _, retriever = indexed
    orchestrator = Orchestrator.from_components(
        settings=settings, retriever=retriever, model=model, store=InMemoryConversationStore(), verifier=Verifier(), policy=VerificationPolicy()
    )
    return TestClient(create_app(settings=settings, orchestrator=orchestrator, retriever=retriever))


def test_health(indexed):
    body = _client(indexed, FakeChatModel({})).get("/health").json()
    assert body["status"] == "ok" and body["collection"] == "legal_it_fake_64" and body["chunks"] > 5
    assert body["llm"] == "ollama:qwen3:4b" and body["embedder"] == "fake:64"


def test_search_endpoint(indexed):
    client = _client(indexed, FakeChatModel({}))
    body = client.post("/v1/search", json={"query": "민법 제750조", "k": 3}).json()
    assert body["hits"][0]["chunk"]["metadata"]["article"] == "제750조"
    filtered = client.post("/v1/search", json={"query": "보증금", "k": 3, "filters": {"document_types": ["court_decision"]}}).json()
    assert {h["chunk"]["document_type"] for h in filtered["hits"]} == {"court_decision"}
    assert client.post("/v1/search", json={"query": "", "k": 3}).status_code == 422
    # Every hit already carries court/law metadata here, so nothing needs a derived label.
    assert body["labels"] == {}


def test_document_endpoint_returns_the_whole_document_and_where_the_chunk_lies(indexed):
    client = _client(indexed, FakeChatModel({}))
    hit = client.post("/v1/search", json={"query": "민법 제750조", "k": 1}).json()["hits"][0]["chunk"]

    body = client.get("/v1/document", params={"chunk_id": hit["id"]}).json()

    assert body["document_id"] == hit["document_id"]
    assert body["text"][body["cited_start"]:body["cited_end"]] == hit["text"]
    assert len(body["text"]) > len(hit["text"])
    assert client.get("/v1/document", params={"chunk_id": "no-such-chunk"}).status_code == 404


def test_a_law_is_read_article_by_article(indexed):
    client = _client(indexed, FakeChatModel({}))

    law = client.get("/v1/law", params={"name": "주택임대차보호법"}).json()
    by_number = {article["article"]: article for article in law["articles"]}
    assert law["name"] == "주택임대차보호법" and by_number["제3조"]["title"] == "대항력 등"
    assert "주민등록을 마친 때" in by_number["제3조"]["text"]
    # The first chunk of an article opens the whole law in the reader.
    whole = client.get("/v1/document", params={"chunk_id": by_number["제3조"]["chunk_id"]}).json()
    assert by_number["제3조"]["text"].split("\n")[0] in whole["text"]
    assert client.get("/v1/law", params={"name": "없는법"}).status_code == 404


def test_cases_are_listed_once_each_with_the_articles_their_text_cites(indexed):
    client = _client(indexed, FakeChatModel({}))

    body = client.post("/v1/cases", json={"query": "임대차 보증금 반환", "k": 10}).json()

    documents = [hit["chunk"]["document_id"] for hit in body["hits"]]
    assert documents and len(documents) == len(set(documents))
    assert {hit["chunk"]["document_type"] for hit in body["hits"]} <= {"court_decision", "judgment", "decision"}
    assert set(body["statutes"]) == set(documents)
    cited = {(ref["law"], ref["article"]) for refs in body["statutes"].values() for ref in refs}
    assert cited and {law for law, _ in cited} <= {"주택임대차보호법", "민법"}
    assert client.post("/v1/cases", json={"query": ""}).status_code == 422


def test_chat_endpoint_round_trip(indexed):
    # A citation-free answer verifies against any evidence, so the script cannot run dry
    # whichever way the verifier decides on the grounded answer.
    plain = LegalAnswer(summary="일반적인 안내입니다.", analysis="일반적인 절차만 설명합니다.", confidence=Confidence.low)
    model = FakeChatModel({
        "QueryAnalysis": [QueryAnalysis(issues=["보증금 반환"], keywords=["임차권등기명령"]), QueryAnalysis(), QueryAnalysis()],
        "LegalAnswer": [_answer(), plain, plain, plain, plain, plain, plain],
    })
    client = _client(indexed, model)
    body = client.post("/v1/chat", json={"message": "임차권등기명령 신청서에 적어야 하는 사항은?"}).json()
    assert body["kind"] == "answer" and body["conversation_id"]
    assert body["evidence"]["items"][0]["source_id"] == "S1" and body["verification"]["ok"] is True
    assert body["answer"]["citations"][0]["source_id"] == "S1"
    assert "trace" not in body
    chosen = client.post("/v1/search", json={"query": "손해배상 책임", "k": 10}).json()["hits"][-1]["chunk"]["id"]
    asked = client.post("/v1/chat", json={"message": "이 판례가 제 사건에도 적용될 수 있나요?", "pinned_chunk_ids": [chosen, "no-such-chunk"]}).json()
    assert asked["evidence"]["items"][0]["chunk_id"] == chosen
    assert client.post("/v1/chat", json={"message": "x", "pinned_chunk_ids": ["a", "b", "c"]}).status_code == 422
    traced = client.post("/v1/chat", json={"message": "후속 질문", "conversation_id": body["conversation_id"], "include_trace": True}).json()
    assert traced["trace"]["policy_action"] in {"accept", "strip"} and traced["conversation_id"] == body["conversation_id"]


def test_chat_stream_sends_the_draft_then_the_same_body_as_chat(indexed):
    def client():
        return _client(indexed, FakeChatModel({"QueryAnalysis": [QueryAnalysis(issues=["보증금 반환"])], "LegalAnswer": [_answer()]}))

    message = {"message": "임차권등기명령 신청서에 적어야 하는 사항은?", "conversation_id": "s1"}
    with client().stream("POST", "/v1/chat/stream", json=message) as response:
        assert response.headers["content-type"].startswith("application/x-ndjson")
        events = [json.loads(line) for line in response.iter_lines() if line]
    assert events[0] == {"type": "draft_reset"} and events[-1]["type"] == "response"
    body = events[-1]["body"]
    assert "".join(event["text"] for event in events if event["type"] == "draft") == body["answer"]["analysis"]
    assert body == client().post("/v1/chat", json=message).json()


def test_a_failed_stream_turn_ends_with_an_error_line(indexed):
    # The model has no answer scripted, so the turn raises inside the worker thread once generation starts.
    with _client(indexed, FakeChatModel({})).stream("POST", "/v1/chat/stream", json={"message": "x"}) as response:
        assert [json.loads(line) for line in response.iter_lines() if line] == [{"type": "draft_reset"}, {"type": "error"}]


def test_chat_validation_error_needs_no_backends():
    settings = load_settings(profile="default", overrides={"embedding": {"backend": "fake"}, "llm": {"backend": "fake"}})
    client = TestClient(create_app(settings=settings))
    assert client.post("/v1/chat", json={}).status_code == 422
    assert client.post("/v1/chat", json={"message": "x", "extra": 1}).status_code == 422
    assert client.get("/health").json()["collection"] is None

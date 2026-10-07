"""The service boundary: only the web service, holding the shared token, may call /v1, and every
conversation is kept under the workspace the web service names."""

import json

from fastapi.testclient import TestClient

from legal_rag.api import create_app
from legal_rag.api.security import conversation_key
from legal_rag.llm import FakeChatModel
from legal_rag.rag.conversation import InMemoryConversationStore
from legal_rag.schemas import Confidence, LegalAnswer, QueryAnalysis
from tests.legal_rag.integration.test_api import SERVICE_TOKEN, WORKSPACE, _client, auth, secured

OTHER = "workspace-bbbbbbbbbbbb"
PLAIN = LegalAnswer(summary="일반적인 안내입니다.", analysis="일반적인 절차만 설명합니다.", confidence=Confidence.low)


def _model(turns=4):
    return FakeChatModel({"QueryAnalysis": [QueryAnalysis() for _ in range(turns * 3)],
                          "LegalAnswer": [PLAIN for _ in range(turns * 7)]})


def test_without_a_configured_token_every_v1_route_is_closed_and_health_stays_open(indexed):
    settings, _, retriever = indexed
    client = TestClient(create_app(settings=settings, retriever=retriever), headers=auth())
    assert client.post("/v1/search", json={"query": "보증금", "k": 1}).status_code == 503
    assert client.get("/v1/law", params={"name": "민법"}).status_code == 503
    assert client.get("/health").status_code == 200


def test_a_caller_needs_the_token_and_a_well_formed_workspace(indexed):
    client = _client(indexed, FakeChatModel({}))
    search = {"query": "보증금", "k": 1}
    assert client.post("/v1/search", json=search).status_code == 200
    assert client.post("/v1/search", json=search, headers={"Authorization": "Bearer " + "x" * 40}).status_code == 401
    assert client.post("/v1/search", json=search, headers={"X-Workspace-ID": ""}).status_code == 400
    assert client.post("/v1/search", json=search, headers={"X-Workspace-ID": "../../other-workspace"}).status_code == 400
    for path, params in (("/v1/document", {"chunk_id": "x"}), ("/v1/law", {"name": "민법"})):
        assert client.get(path, params=params, headers={"Authorization": ""}).status_code == 401
    assert client.post("/v1/cases", json={"query": "보증금"}, headers={"Authorization": ""}).status_code == 401
    assert client.delete("/v1/workspace/conversations", headers={"Authorization": ""}).status_code == 401


def test_the_same_conversation_id_in_two_workspaces_keeps_two_histories(indexed):
    store = InMemoryConversationStore()
    client = _client(indexed, _model(), store=store)
    message = {"message": "보증금을 못 받았어요", "conversation_id": "c1"}
    assert client.post("/v1/chat", json=message).json()["conversation_id"] == "c1"
    assert client.post("/v1/chat", json=message, headers={"X-Workspace-ID": OTHER}).json()["conversation_id"] == "c1"
    mine, theirs = (conversation_key(owner, "c1") for owner in _owners(client, WORKSPACE, OTHER))
    assert mine != theirs
    assert len(store.get(mine).turns) == 2 and len(store.get(theirs).turns) == 2

    assert client.delete("/v1/workspace/conversations").json() == {"ok": True}
    assert store.get(mine).turns == [] and len(store.get(theirs).turns) == 2


def test_the_stream_route_keys_turns_by_workspace_and_frees_its_slot(indexed):
    store = InMemoryConversationStore()
    client = _client(indexed, _model(), store=store)
    for _ in range(2):  # one admission slot: a slot kept after the first turn would refuse the second (429)
        with client.stream("POST", "/v1/chat/stream", json={"message": "보증금 반환", "conversation_id": "s1"}) as response:
            assert response.status_code == 200
            events = [json.loads(line) for line in response.iter_lines() if line]
        assert events[-1]["type"] == "response" and events[-1]["body"]["conversation_id"] == "s1"
    (owner,) = _owners(client, WORKSPACE)
    assert len(store.get(conversation_key(owner, "s1")).turns) == 4
    assert store.get("s1").turns == []


def test_traces_are_sent_only_when_the_operator_allows_them(indexed):
    client = _client(indexed, _model(), allow_trace=False)
    assert client.post("/v1/chat", json={"message": "보증금", "include_trace": True}).status_code == 403
    assert client.post("/v1/chat", json={"message": "보증금"}).status_code == 200


def test_the_body_limit_admits_the_largest_valid_chat_and_refuses_more(indexed):
    client = _client(indexed, _model())
    # ASCII JSON escapes every Hangul syllable as \uXXXX: six bytes a character.
    largest = {"message": "가" * 4000, "user_documents": [{"name": "계약서", "text": "나" * 4000} for _ in range(3)]}
    body = json.dumps(largest, ensure_ascii=True).encode()
    assert len(body) > 90_000
    sent = client.post("/v1/chat", content=body, headers={"Content-Type": "application/json"})
    assert sent.status_code != 413
    assert client.post("/v1/chat", content=b"{" + b" " * 200_000 + b"}",
                       headers={"Content-Type": "application/json"}).status_code == 413


def test_a_workspace_is_rate_limited(indexed):
    client = _client(indexed, FakeChatModel({}), requests_per_minute=2)
    search = {"query": "보증금", "k": 1}
    assert [client.post("/v1/search", json=search).status_code for _ in range(3)] == [200, 200, 429]
    assert client.post("/v1/search", json=search, headers={"X-Workspace-ID": OTHER}).status_code == 200


def test_a_full_workspace_is_refused_not_evicted(indexed, monkeypatch):
    store = InMemoryConversationStore()
    monkeypatch.setattr(store, "has_capacity", lambda prefix, needed=2: False)
    client = _client(indexed, _model(), store=store)
    assert client.post("/v1/chat", json={"message": "보증금"}).status_code == 429


def test_remote_models_are_refused_unless_allowed(indexed):
    settings, _, retriever = indexed
    remote = secured(settings.model_copy(update={"llm": settings.llm.model_copy(update={"base_url": "https://models.example"})}))
    client = TestClient(create_app(settings=remote, retriever=retriever), headers=auth())
    assert client.post("/v1/search", json={"query": "보증금", "k": 1}).status_code == 503
    plain_http = secured(settings.model_copy(update={"llm": settings.llm.model_copy(update={"base_url": "http://models.example"})}),
                         allow_external_models=True)
    client = TestClient(create_app(settings=plain_http, retriever=retriever), headers=auth())
    assert client.post("/v1/search", json={"query": "보증금", "k": 1}).json() == {"detail": "model_tls_required"}


def _owners(client, *workspaces):
    """The store prefix the service derives for each workspace header."""
    security = client.app.state.security
    return [security.authenticate(_Request(workspace)) for workspace in workspaces]


class _Request:
    def __init__(self, workspace):
        self.headers = {"Authorization": f"Bearer {SERVICE_TOKEN}", "X-Workspace-ID": workspace}

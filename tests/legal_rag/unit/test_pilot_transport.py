import pytest
from pydantic import ValidationError
from legal_rag.api.models import ChatRequest
from legal_rag.schemas import ChatResponse


def test_withheld_response_round_trip():
    body = ChatResponse.model_validate({"kind": "withheld", "conversation_id": "c", "withheld": {
        "reason_code": "insufficient_evidence", "known_facts": ["가상 계약"],
        "unresolved_items": ["수리 범위"], "explanation": "결론을 보류합니다."}})
    assert body.model_dump()["withheld"]["known_facts"] == ["가상 계약"]
    assert body.suppress_followup_questions is False


def test_cli_renders_withheld_without_answer_or_sources(capsys):
    from scripts.chat import render
    response = ChatResponse(kind="withheld", conversation_id="c", withheld={
        "reason_code": "evidence_gap", "known_facts": ["가상 계약"],
        "unresolved_items": ["미확인 사실"], "explanation": "판단을 보류합니다."})
    render(response, False)
    text = capsys.readouterr().out
    assert all(part in text for part in ("판단을 보류합니다.", "evidence_gap", "가상 계약", "미확인 사실"))
    assert "[인용]" not in text


def test_scope_is_explicit_frozen_and_requires_request_identity():
    from legal_rag.schemas.action import PolicyScope
    scope = PolicyScope(matter_id="fictional-lease-1", issue_id="rent-deduction")
    with pytest.raises(ValidationError):
        scope.matter_id = "changed"
    for request_id in (None, "", "  "):
        with pytest.raises(ValidationError):
            ChatRequest(message="질문", policy_scope=scope, request_id=request_id)
    assert ChatRequest(message="질문").policy_scope is None
    assert ChatRequest(message="질문", policy_scope=scope, request_id="r1", conversation_id="c").request_id == "r1"
    with pytest.raises(ValidationError):
        PolicyScope(matter_id="x", issue_id="unknown")


@pytest.mark.parametrize("endpoint", ["/v1/chat", "/v1/chat/stream"])
def test_api_forwards_scope_only_when_opted_in(endpoint):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from legal_rag.api.routes import router
    from types import SimpleNamespace
    calls = []
    def handle(message, **kwargs):
        calls.append(kwargs)
        return ChatResponse(kind="rejected", conversation_id="c", reason="test")
    app = FastAPI()
    app.include_router(router)
    app.state.get_orchestrator = lambda: SimpleNamespace(handle=handle)
    client = TestClient(app)
    assert client.post(endpoint, json={"message": "q"}).status_code == 200
    assert "policy_scope" not in calls[-1] and "request_id" not in calls[-1]
    scope = {"matter_id": "fictional-lease-1", "issue_id": "rent-deduction"}
    response = client.post(endpoint, json={"message": "q", "policy_scope": scope, "request_id": "r1", "conversation_id": "c"})
    assert response.status_code == 200
    assert calls[-1]["policy_scope"].matter_id == "fictional-lease-1"
    assert calls[-1]["request_id"] == "r1"


@pytest.mark.parametrize("before_handle", [False, True])
@pytest.mark.parametrize("endpoint", ["/v1/chat", "/v1/chat/stream"])
def test_api_policy_conflicts_are_safe(endpoint, before_handle):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from legal_rag.api.routes import router
    from legal_rag.rag.policy_state import RequestConflict
    from types import SimpleNamespace
    def fail(*args, **kwargs):
        raise RequestConflict("secret raw facts")
    app = FastAPI()
    app.include_router(router)
    app.state.get_orchestrator = fail if before_handle else lambda: SimpleNamespace(handle=fail)
    response = TestClient(app, raise_server_exceptions=False).post(endpoint, json={"message": "q"})
    assert "secret raw facts" not in response.text
    if endpoint.endswith("/stream") and not before_handle:
        assert response.status_code == 200
        assert response.json() == {"type": "error", "reason": "policy_request_conflict"}
    else:
        assert response.status_code == 409


@pytest.mark.parametrize("conversation_id", [None, "", "  "])
def test_pilot_requires_stable_conversation_identity_for_replay(conversation_id):
    scope = {"matter_id": "fictional-lease-1", "issue_id": "rent-deduction"}
    with pytest.raises(ValidationError):
        ChatRequest(message="q", policy_scope=scope, request_id="r1", conversation_id=conversation_id)

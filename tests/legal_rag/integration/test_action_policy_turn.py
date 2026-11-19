"""Controlled end-to-end policy execution, not legal correctness evaluation."""

import pytest

from legal_rag.llm import FakeChatModel
from legal_rag.schemas import QueryAnalysis, UserDocument, StandaloneQuestion
from legal_rag.rag.sufficiency import SufficiencyAssessment, FactFinding, EvidenceFinding, FactObservation, LegalSupport
from tests.legal_rag.integration.test_chat_turn import _orchestrator, _answer


SCOPE = {"profile": "lease_review_v1", "matter_id": "fictional-lease-1", "issue_id": "rent-deduction"}
QUESTION = "중도 퇴거 때 이 특약에 따라 월세를 보증금에서 공제할 수 있나요?"
DOCUMENT = UserDocument(name="가상계약서", text="임차인이 중도 퇴거하면 새 임차인이 들어올 때까지 차임을 부담한다.")


def missing_fact():
    return SufficiencyAssessment(
        facts=[FactFinding(key="termination_agreement", status="missing", reason="이후 합의 여부가 제시되지 않음")],
        evidence=[EvidenceFinding(key="deduction_basis", proposition="종료 합의와 차임 공제의 관계",
                                  status="insufficient", reason="결정적인 근거를 추가 확인해야 함")],
    )


def test_pilot_searches_before_asking_exactly_one_fact(retriever):
    settings, r = retriever
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis(needs_more_info=True, missing_facts=["여러 사실"] )],
                           "SufficiencyAssessment": [missing_fact()]})
    orch = _orchestrator(settings, r, model)
    response = orch.handle(QUESTION, conversation_id="pilot1", user_documents=[DOCUMENT],
                           policy_scope=SCOPE, request_id="first")
    assert response.kind == "clarification"
    assert len(response.questions) == 1
    assert response.trace.evidence_ids
    assert [d.action for d in response.trace.post_retrieval] == ["clarify"]
    assert not any(c.response_schema.__name__ == "LegalAnswer" for c in model.calls)


def test_same_issue_cannot_ask_again_after_user_does_not_know(retriever):
    settings, r = retriever
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis(), QueryAnalysis()],
                           "StandaloneQuestion": [StandaloneQuestion(question=QUESTION)],
                           "SufficiencyAssessment": [missing_fact(), missing_fact()]})
    orch = _orchestrator(settings, r, model)
    first = orch.handle(QUESTION, conversation_id="pilot2", user_documents=[DOCUMENT],
                        policy_scope=SCOPE, request_id="first")
    second = orch.handle("모르겠어요", conversation_id="pilot2", user_documents=[DOCUMENT],
                         policy_scope=SCOPE, request_id="second")
    assert first.kind == "clarification"
    assert second.kind == "withheld"
    assert second.withheld.reason_code == "facts_unresolved"
    assert second.questions == []
    assert not any(c.response_schema.__name__ == "LegalAnswer" for c in model.calls)


def test_identical_request_replays_without_another_search_or_model_call(retriever):
    settings, r = retriever
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis()], "SufficiencyAssessment": [missing_fact()]})
    orch = _orchestrator(settings, r, model)
    first = orch.handle(QUESTION, conversation_id="pilot3", user_documents=[DOCUMENT],
                        policy_scope=SCOPE, request_id="same")
    calls = len(model.calls)
    replay = orch.handle(QUESTION, conversation_id="pilot3", user_documents=[DOCUMENT],
                         policy_scope=SCOPE, request_id="same")
    assert replay == first
    assert len(model.calls) == calls


class SequenceRetriever:
    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = []

    def retrieve(self, query, k, filters=None):
        self.calls.append((query, k, filters))
        return self.batches.pop(0)


def known_assessment(hits=()):
    return SufficiencyAssessment(
        facts=[FactFinding(key="termination_agreement", status="known", value="합의함", reason="사용자 진술",
                           observations=[FactObservation(locator="M0", quote="합의했습니다")])],
        evidence=[EvidenceFinding(key="deduction_basis", proposition="합의 종료 뒤 차임 공제의 법률 근거",
                                  status="supported" if hits else "insufficient", reason="통제된 원문 지원 라벨",
                                  supports=[LegalSupport(source_id="S1", quote=hits[0].chunk.text[:20])] if hits else [])],
    )


def test_weak_evidence_gets_one_research_then_verified_answer(retriever):
    settings, r = retriever
    hits = r.retrieve("임차권등기명령 신청서에 적어야 하는 사항", settings.retrieval.k_final)
    sequence = SequenceRetriever([[], hits])
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis()],
                           "SufficiencyAssessment": [known_assessment(), known_assessment(hits)],
                           "LegalAnswer": [_answer().model_copy(update={"additional_information_needed": ["합의했나요?"]})]})
    orch = _orchestrator(settings, sequence, model)
    response = orch.handle(QUESTION + " 합의했습니다", conversation_id="pilot4", user_documents=[DOCUMENT],
                           policy_scope=SCOPE, request_id="research")
    assert response.kind == "answer"
    assert response.verification.ok
    assert len(sequence.calls) == 2
    assert "합의 종료 뒤 차임 공제" in sequence.calls[1][0]
    assert [d.action for d in response.trace.post_retrieval] == ["research", "answer"]
    # The role of every cited case passage is kept for evaluation; statutes have none.
    assert response.trace.post_retrieval[0].support_roles == {}
    cited_case = hits[0].chunk.document_type.value != "statute"
    assert set(response.trace.post_retrieval[1].support_roles) == ({"S1"} if cited_case else set())
    assert response.answer.additional_information_needed == []
    assert response.suppress_followup_questions
    prompts = [c for c in model.calls if c.response_schema.__name__ == "LegalAnswer"]
    assert "후속 질문" in "\n".join(m.content for m in prompts[0].messages)


def test_research_exhaustion_withholds_without_generation(retriever):
    settings, _ = retriever
    sequence = SequenceRetriever([[], []])
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis()],
                           "SufficiencyAssessment": [known_assessment(), known_assessment()]})
    response = _orchestrator(settings, sequence, model).handle(
        QUESTION + " 합의했습니다", conversation_id="pilot5", user_documents=[DOCUMENT],
        policy_scope=SCOPE, request_id="research",
    )
    assert response.kind == "withheld"
    assert response.withheld.reason_code == "evidence_unresolved"
    assert response.withheld.known_facts == ["합의했습니다"]
    assert len(sequence.calls) == 2
    assert [d.action for d in response.trace.post_retrieval] == ["research", "withhold"]


def test_search_failure_is_not_reported_as_insufficient_legal_evidence(retriever):
    from legal_rag.rag.policy_state import RequestFailed
    settings, _ = retriever
    class BrokenRetriever:
        def retrieve(self, query, k, filters=None):
            raise TimeoutError("test search timeout")
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis()]})
    orch = _orchestrator(settings, BrokenRetriever(), model)
    with pytest.raises(TimeoutError, match="test search timeout"):
        orch.handle(QUESTION, conversation_id="pilot6", policy_scope=SCOPE, request_id="error")
    with pytest.raises(RequestFailed):
        orch.handle(QUESTION, conversation_id="pilot6", policy_scope=SCOPE, request_id="error")


def test_saved_question_budget_survives_orchestrator_recreation(retriever, tmp_path):
    from legal_rag.rag.conversation import SQLiteConversationStore
    settings, r = retriever
    path = tmp_path / "conversation.db"
    first_model = FakeChatModel({"QueryAnalysis": [QueryAnalysis()], "SufficiencyAssessment": [missing_fact()]})
    first = _orchestrator(settings, r, first_model, SQLiteConversationStore(path))
    assert first.handle(QUESTION, conversation_id="persist", policy_scope=SCOPE, request_id="first").kind == "clarification"
    second_model = FakeChatModel({"QueryAnalysis": [QueryAnalysis()], "SufficiencyAssessment": [missing_fact()],
                                  "StandaloneQuestion": [StandaloneQuestion(question=QUESTION)]})
    second = _orchestrator(settings, r, second_model, SQLiteConversationStore(path))
    response = second.handle("잘 모르겠어요", conversation_id="persist", policy_scope=SCOPE, request_id="second")
    assert response.kind == "withheld"
    assert response.withheld.reason_code == "facts_unresolved"


def test_different_matter_has_separate_question_budget_and_history(retriever):
    settings, r = retriever
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis(), QueryAnalysis()],
                           "SufficiencyAssessment": [missing_fact(), missing_fact()]})
    orch = _orchestrator(settings, r, model)
    first = orch.handle(QUESTION, conversation_id="matters", policy_scope=SCOPE, request_id="first")
    second = orch.handle(QUESTION, conversation_id="matters", policy_scope={**SCOPE, "matter_id": "fictional-lease-2"},
                         request_id="second")
    assert first.kind == second.kind == "clarification"
    assert not any(c.response_schema.__name__ == "StandaloneQuestion" for c in model.calls)


def test_invalid_assessment_withholds_without_asking_user_or_generating(retriever):
    settings, r = retriever
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis()], "SufficiencyAssessment": ["not json"]})
    response = _orchestrator(settings, r, model).handle(QUESTION, conversation_id="invalid",
                                                       policy_scope=SCOPE, request_id="first")
    assert response.kind == "withheld"
    assert response.withheld.reason_code == "assessment_failed"
    assert response.questions == []





def test_new_fact_after_clarification_allows_answer_without_a_second_question(retriever):
    settings, r = retriever
    hits = r.retrieve("임차권등기명령 신청서에 적어야 하는 사항", settings.retrieval.k_final)
    sequence = SequenceRetriever([hits, hits])
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis(), QueryAnalysis()],
                           "StandaloneQuestion": [StandaloneQuestion(question="임차권등기명령 신청서에 적어야 하는 사항")],
                           "SufficiencyAssessment": [missing_fact(), known_assessment(hits)],
                           "LegalAnswer": [_answer()]})
    orch = _orchestrator(settings, sequence, model)
    assert orch.handle(QUESTION, conversation_id="resolved", policy_scope=SCOPE, request_id="first").kind == "clarification"
    response = orch.handle("종료에 합의했습니다", conversation_id="resolved", policy_scope=SCOPE, request_id="second")
    assert response.kind == "answer"
    assert len(sequence.calls) == 2
    assert response.trace.post_retrieval[0].action == "answer"


@pytest.mark.parametrize("streaming", [False, True])
def test_real_api_pipeline_preserves_withheld_reason_and_replay(retriever, streaming):
    import json
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from legal_rag.api.routes import router
    settings, r = retriever
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis(), QueryAnalysis()],
                           "StandaloneQuestion": [StandaloneQuestion(question=QUESTION)],
                           "SufficiencyAssessment": [missing_fact(), missing_fact()]})
    orch = _orchestrator(settings, r, model)
    app = FastAPI()
    app.include_router(router)
    app.state.get_orchestrator = lambda: orch
    endpoint = "/v1/chat/stream" if streaming else "/v1/chat"
    with TestClient(app) as client:
        def send(message, request_id):
            response = client.post(endpoint, json={"message": message, "conversation_id": "api-pilot",
                "policy_scope": SCOPE, "request_id": request_id, "user_documents": [DOCUMENT.model_dump()]})
            assert response.status_code == 200
            if streaming:
                events = [json.loads(line) for line in response.text.splitlines()]
                assert not any(event["type"] == "draft" for event in events)
                return next(event["body"] for event in events if event["type"] == "response")
            return response.json()
        assert send(QUESTION, "first")["kind"] == "clarification"
        second = send("모르겠어요", "second")
        assert second["kind"] == "withheld"
        assert second["withheld"]["reason_code"] == "facts_unresolved"
        assert "trace" not in second
        calls = len(model.calls)
        assert send("모르겠어요", "second") == second
        assert len(model.calls) == calls

import pytest

from legal_rag.llm import FakeChatModel
from legal_rag.rag.conversation import InMemoryConversationStore
from legal_rag.rag.orchestrator import Orchestrator
from legal_rag.schemas import Citation, CitationKind, Confidence, LegalAnswer, QueryAnalysis
from legal_rag.verification import VerificationPolicy, Verifier
from tests.legal_rag.integration.test_ingest_to_search import indexed  # noqa: F401


def _answer(source_id="S1", reference="주택임대차보호법 제3조의3", analysis="주택임대차보호법 제3조의3에 따라 임차권등기명령을 신청할 수 있습니다."):
    return LegalAnswer(
        summary="보증금 반환을 청구할 수 있습니다.", legal_issues=["보증금 반환"], analysis=analysis,
        possible_actions=["임차권등기명령 신청"], additional_information_needed=[],
        citations=[Citation(source_id=source_id, kind=CitationKind.statute, reference=reference)],
        uncertainty="", confidence=Confidence.medium,
    )


def _orchestrator(settings, retriever, model, store=None):
    return Orchestrator.from_components(
        settings=settings, retriever=retriever, model=model, store=store or InMemoryConversationStore(),
        verifier=Verifier(), policy=VerificationPolicy(max_regenerations=settings.rag.max_regenerations),
    )


@pytest.fixture
def retriever(indexed):
    settings, _, hybrid = indexed
    return settings, hybrid


def test_grounded_turn_returns_answer_with_evidence(retriever):
    settings, r = retriever
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis(domain="민사", issues=["보증금 반환"], keywords=["임차권등기명령"])], "LegalAnswer": [_answer()]})
    resp = _orchestrator(settings, r, model).handle("임차권등기명령 신청서에 적어야 하는 사항은?", conversation_id="c1")
    assert resp.kind == "answer" and resp.verification.ok
    assert resp.evidence.items[0].source_id == "S1"
    assert any(e.header.get("Article") == "제3조의3" for e in resp.evidence.items)
    assert resp.trace.timings and resp.trace.policy_action == "accept" and resp.trace.analysis.domain == "민사"
    assert "fused" in resp.trace.retrieved


def test_hallucinated_citation_triggers_regeneration_then_strip(retriever):
    settings, r = retriever
    bad = _answer(source_id="S9", reference="민법 제839조의2", analysis="민법 제839조의2에 따라 청구할 수 있습니다. 보증금은 반환되어야 합니다.")
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis(issues=["보증금 반환"])], "LegalAnswer": [bad, bad]})
    resp = _orchestrator(settings, r, model).handle("보증금을 못 받았어요", conversation_id="c2")
    assert resp.kind == "answer" and resp.trace.regenerations == 1 and resp.trace.policy_action == "strip"
    assert resp.answer.confidence == Confidence.low and resp.answer.citations == []
    assert "제839조의2" not in resp.answer.analysis and "보증금은 반환되어야 합니다." in resp.answer.analysis
    generation_calls = [c for c in model.calls if c.response_schema.__name__ == "LegalAnswer"]
    assert len(generation_calls) == 2 and "근거 없는 인용" in generation_calls[1].messages[-1].content


def test_stripped_answer_with_nothing_left_is_rejected(retriever):
    settings, r = retriever
    bad = _answer(source_id="S9", reference="민법 제839조의2", analysis="민법 제839조의2에 따라 청구할 수 있습니다.")
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis()], "LegalAnswer": [bad, bad]})
    resp = _orchestrator(settings, r, model).handle("재산분할", conversation_id="c5")
    assert resp.kind == "rejected" and resp.answer is None and resp.trace.policy_action == "reject"


def test_missing_facts_returns_clarification_without_retrieval(retriever):
    settings, r = retriever
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis(needs_more_info=True, missing_facts=["임대차 계약 종료일", "보증금 액수"])]})
    resp = _orchestrator(settings, r, model).handle("법적으로 어떻게 되나요?", conversation_id="c3")
    assert resp.kind == "clarification" and resp.questions == ["임대차 계약 종료일", "보증금 액수"] and resp.evidence is None
    assert resp.trace.policy_action == "clarify" and resp.trace.search_queries == []


def test_history_is_stored_but_evidence_text_is_not(retriever):
    settings, r = retriever
    store = InMemoryConversationStore()
    plain = LegalAnswer(summary="일반적인 안내입니다.", analysis="근거 없이 일반적인 절차만 설명합니다.", confidence=Confidence.low)
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis(), QueryAnalysis()], "LegalAnswer": [plain, plain]})
    orch = _orchestrator(settings, r, model, store)
    orch.handle("질문", conversation_id="c4")
    orch.handle("후속 질문", conversation_id="c4")
    turns = store.get("c4").turns
    assert [t.role for t in turns] == ["user", "assistant", "user", "assistant"]
    assert turns[1].evidence_ids and "LEGAL SOURCE" not in turns[1].content
    assert "질문" in model.calls[2].messages[-1].content and "이전 대화" in model.calls[2].messages[-1].content


def _capture_filters(monkeypatch, hybrid):
    seen = []
    original = hybrid.retrieve_many

    def spy(queries, k, filters=None, **kwargs):
        seen.append(filters)
        return original(queries, k, filters, **kwargs)

    monkeypatch.setattr(hybrid, "retrieve_many", spy)
    return seen


def test_analyzer_document_type_is_not_a_hard_filter_by_default(retriever, monkeypatch):
    """On 210 eval questions the analyzer asked for statutes 201 times; filtering on it cut
    gold Success@10 from 0.557 to 0.414 (95% CI of the change [-0.205, -0.086])."""
    settings, r = retriever
    seen = _capture_filters(monkeypatch, r)
    analysis = QueryAnalysis(issues=["보증금 반환"], document_types_needed=["statute"])
    model = FakeChatModel({"QueryAnalysis": [analysis], "LegalAnswer": [_answer()]})
    _orchestrator(settings, r, model).handle("보증금 반환 요건은?", conversation_id="f1")
    assert seen == [None]


def test_hard_filter_can_be_enabled(retriever, monkeypatch):
    settings, r = retriever
    settings = settings.model_copy(update={"rag": settings.rag.model_copy(update={"hard_filter": True})})
    seen = _capture_filters(monkeypatch, r)
    analysis = QueryAnalysis(issues=["보증금 반환"], document_types_needed=["statute"])
    # Filtered evidence may fail verification and regenerate; only the filter is under test.
    model = FakeChatModel({"QueryAnalysis": [analysis], "LegalAnswer": [_answer(), _answer()]})
    _orchestrator(settings, r, model).handle("보증금 반환 요건은?", conversation_id="f2")
    assert seen[0] is not None and seen[0].citation_kind == "statute"


_TRUNCATED = '{"summary": "보증금 반환을 청구할 수 있습니다.", "analysis": "주택임대차보호법 제3조의3에 따라'


def test_failed_generation_is_a_refusal_not_a_server_error(retriever):
    """A GenerationError on JSON cut off at max_tokens escaped handle() and /v1/chat
    answered 500 on 2 of 80 turns (2026-09-28)."""
    settings, r = retriever
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis()], "LegalAnswer": [_TRUNCATED, _TRUNCATED]})
    resp = _orchestrator(settings, r, model).handle("보증금을 못 받았어요", conversation_id="c6")
    assert resp.kind == "rejected" and resp.reason == "answer generation failed" and resp.answer is None
    assert resp.evidence.items and resp.trace.policy_action == "reject"


def test_failed_regeneration_strips_the_previous_draft(retriever):
    settings, r = retriever
    bad = _answer(source_id="S9", reference="민법 제839조의2", analysis="민법 제839조의2에 따라 청구할 수 있습니다. 보증금은 반환되어야 합니다.")
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis()], "LegalAnswer": [bad, _TRUNCATED, _TRUNCATED]})
    resp = _orchestrator(settings, r, model).handle("보증금을 못 받았어요", conversation_id="c7")
    assert resp.kind == "answer" and resp.trace.policy_action == "strip" and resp.trace.regenerations == 1
    assert "제839조의2" not in resp.answer.analysis and "보증금은 반환되어야 합니다." in resp.answer.analysis

from datetime import datetime, timezone

import pytest

from legal_rag.llm import FakeChatModel
from legal_rag.rag.conversation import InMemoryConversationStore
from legal_rag.rag.orchestrator import Orchestrator
from legal_rag.schemas import Citation, CitationKind, Confidence, LegalAnswer, QueryAnalysis, StandaloneQuestion, Turn
from legal_rag.verification import VerificationPolicy, Verifier


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


def test_missing_facts_with_a_legal_area_are_answered_and_carried_into_the_answer(retriever):
    settings, r = retriever
    analysis = QueryAnalysis(domain="민사", issues=["보증금 반환"], needs_more_info=True, missing_facts=["임대차 계약 종료일", "보증금 액수"])
    model = FakeChatModel({"QueryAnalysis": [analysis], "LegalAnswer": [_answer()]})
    resp = _orchestrator(settings, r, model).handle("집주인이 보증금을 안 돌려줘요", conversation_id="n1")
    assert resp.kind == "answer" and resp.evidence.items
    prompt = _generation_prompts(model)[0]
    assert "아직 확인되지 않은 사실" in prompt and "임대차 계약 종료일" in prompt and "보증금 액수" in prompt
    assert prompt.index("아직 확인되지 않은 사실") < prompt.index("질문: 집주인이")


def test_no_open_questions_block_when_nothing_is_missing(retriever):
    settings, r = retriever
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis(domain="민사")], "LegalAnswer": [_answer()]})
    _orchestrator(settings, r, model).handle("임차권등기명령 신청서에 적어야 하는 사항은?", conversation_id="n2")
    assert "아직 확인되지 않은 사실" not in _generation_prompts(model)[0]


def test_history_is_stored_but_evidence_text_is_not(retriever):
    settings, r = retriever
    store = InMemoryConversationStore()
    plain = LegalAnswer(summary="일반적인 안내입니다.", analysis="근거 없이 일반적인 절차만 설명합니다.", confidence=Confidence.low)
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis(), QueryAnalysis()], "StandaloneQuestion": [StandaloneQuestion(question="질문에 이어 후속 질문")], "LegalAnswer": [plain, plain]})
    orch = _orchestrator(settings, r, model, store)
    orch.handle("질문", conversation_id="c4")
    orch.handle("후속 질문", conversation_id="c4")
    turns = store.get("c4").turns
    assert [t.role for t in turns] == ["user", "assistant", "user", "assistant"]
    assert turns[1].evidence_ids and "LEGAL SOURCE" not in turns[1].content
    assert "질문" in model.calls[2].messages[-1].content and "이전 대화" in model.calls[2].messages[-1].content


def _spy_retrieval(monkeypatch, hybrid):
    seen = []
    original = hybrid.retrieve_many

    def spy(queries, k, filters=None, **kwargs):
        seen.append({"queries": list(queries), "rerank_query": kwargs.get("rerank_query")})
        return original(queries, k, filters, **kwargs)

    monkeypatch.setattr(hybrid, "retrieve_many", spy)
    return seen


def _generation_prompts(model):
    return [" ".join(m.content for m in c.messages) for c in model.calls if c.response_schema.__name__ == "LegalAnswer"]


def test_a_follow_up_is_retrieved_and_answered_as_a_standalone_question(retriever, monkeypatch):
    settings, r = retriever
    seen = _spy_retrieval(monkeypatch, r)
    standalone = "임차권등기명령은 어느 법원에 신청하나요?"
    model = FakeChatModel({
        "QueryAnalysis": [QueryAnalysis(), QueryAnalysis()],
        "StandaloneQuestion": [StandaloneQuestion(question=standalone)],
        "LegalAnswer": [_answer(), _answer()],
    })
    orch = _orchestrator(settings, r, model)
    orch.handle("임차권등기명령 신청서에 적어야 하는 사항은?", conversation_id="m1")
    orch.handle("그건 어느 법원에 하나요?", conversation_id="m1")

    assert [c.response_schema.__name__ for c in model.calls].count("StandaloneQuestion") == 1
    assert seen[1]["queries"][0] == standalone and seen[1]["rerank_query"] == standalone
    first, second = _generation_prompts(model)
    block = "이전 대화(맥락 참고용"  # the history message, not the system prompt's mention of earlier turns
    assert block not in first and "질문: 임차권등기명령 신청서에" in first
    assert "질문: " + standalone in second
    assert block in second and "임차권등기명령 신청서에 적어야 하는 사항은?" in second


def test_a_reply_to_a_clarification_is_merged_with_the_original_question(retriever, monkeypatch):
    settings, r = retriever
    seen = _spy_retrieval(monkeypatch, r)
    merged = "임대차 종료 후 보증금 2천만 원을 못 받았을 때 임차권등기명령을 신청할 수 있나요?"
    model = FakeChatModel({
        "QueryAnalysis": [QueryAnalysis(needs_more_info=True, missing_facts=["보증금 액수"]), QueryAnalysis()],
        "StandaloneQuestion": [StandaloneQuestion(question=merged)],
        "LegalAnswer": [_answer()],
    })
    orch = _orchestrator(settings, r, model)
    assert orch.handle("집주인이 돈을 안 줘요", conversation_id="m2").kind == "clarification"
    assert orch.handle("2천만 원이고 계약은 지난달에 끝났어요", conversation_id="m2").kind == "answer"

    condense_call = next(c for c in model.calls if c.response_schema.__name__ == "StandaloneQuestion")
    assert "집주인이 돈을 안 줘요" in condense_call.messages[-1].content and "보증금 액수" in condense_call.messages[-1].content
    assert seen[0]["queries"][0] == merged


def test_a_new_conversation_never_calls_the_condenser(retriever):
    settings, r = retriever
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis()], "LegalAnswer": [_answer()]})
    _orchestrator(settings, r, model).handle("임차권등기명령 신청서에 적어야 하는 사항은?", conversation_id="m3")
    assert "StandaloneQuestion" not in [c.response_schema.__name__ for c in model.calls]


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


REPAIRS = "민법 제623조에 따라 임대인은 집을 쓸 수 있는 상태로 유지해야 합니다."
REPEATS = REPAIRS + " 주택임대차보호법 제3조의3도 참고하세요."


def _lease_model(*answers):
    return FakeChatModel({"QueryAnalysis": [QueryAnalysis(), QueryAnalysis()], "LegalAnswer": list(answers),
                          "StandaloneQuestion": [StandaloneQuestion(question="임차권등기명령은 어느 법원에 신청하나요?")] * 2})


def test_a_follow_up_may_repeat_what_the_last_answer_cited(retriever):
    """The model reads the last answer in its history; this turn's search need not return that statute again."""
    settings, r = retriever
    store = InMemoryConversationStore()
    for role, content in (("user", "집주인이 보일러를 안 고쳐 줘요"), ("assistant", REPAIRS)):
        store.append("p1", Turn(role=role, content=content, created_at=datetime.now(timezone.utc), evidence_ids=["c1"] * (role == "assistant")))
    other_law = _answer(analysis="상법 제623조에 따라 고쳐야 합니다. 주택임대차보호법 제3조의3도 참고하세요.")
    orch = _orchestrator(settings, r, _lease_model(_answer(analysis=REPEATS), other_law, other_law), store)
    repeated = orch.handle("보증금도 못 받았어요", conversation_id="p1")
    assert repeated.trace.policy_action == "accept" and repeated.prior_refs == ["민법 제623조"]
    assert orch.handle("어느 법원인가요?", conversation_id="p1").trace.policy_action == "strip"


def test_a_turn_that_lost_its_citation_passes_nothing_on(retriever):
    settings, r = retriever
    unsupported = LegalAnswer(summary="수선을 요구할 수 있습니다.", analysis=REPAIRS, confidence=Confidence.medium)
    orch = _orchestrator(settings, r, _lease_model(unsupported, unsupported, _answer(analysis=REPEATS), _answer(analysis=REPEATS)))
    first = orch.handle("집주인이 보일러를 안 고쳐 줘요", conversation_id="p2")
    second = orch.handle("보증금도 못 받았어요", conversation_id="p2")
    assert first.prior_refs == [] and first.trace.policy_action != "accept"
    assert second.prior_refs == [] and second.trace.policy_action == "strip"


def test_a_clarifying_question_passes_nothing_on(retriever):
    """The analyzer writes the question; a statute it names there was never checked against evidence."""
    settings, r = retriever
    asks = QueryAnalysis(needs_more_info=True, missing_facts=["민법 제623조의 수선을 집주인에게 요구했나요?"])
    model = FakeChatModel({"QueryAnalysis": [asks, QueryAnalysis()], "LegalAnswer": [_answer(analysis=REPEATS)] * 2,
                           "StandaloneQuestion": [StandaloneQuestion(question="임차권등기명령은 어느 법원에 신청하나요?")]})
    orch = _orchestrator(settings, r, model)
    assert orch.handle("집 문제예요", conversation_id="p3").kind == "clarification"
    answered = orch.handle("보증금도 못 받았어요", conversation_id="p3")
    assert answered.prior_refs == [] and answered.trace.policy_action == "strip"


def test_a_chosen_passage_leads_the_evidence_and_keeps_its_length(retriever):
    """A user who asks about one case gets that case in the evidence, whatever retrieval ranks."""
    settings, r = retriever
    case = next(hit.chunk for hit in r.retrieve("손해배상 책임", 10, None) if hit.chunk.document_type.value != "statute")
    # An answer without citations verifies against any evidence, so the script does not depend on its order.
    plain = LegalAnswer(summary="일반적인 안내입니다.", analysis="일반적인 절차만 설명합니다.", confidence=Confidence.low)
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis(issues=["보증금 반환"]), QueryAnalysis(issues=["보증금 반환"])],
                           "LegalAnswer": [plain] * 6})
    orchestrator = _orchestrator(settings, r, model)
    question = "임차권등기명령 신청서에 적어야 하는 사항은?"

    plain = orchestrator.handle(question, conversation_id="p1")
    pinned = orchestrator.handle(question, conversation_id="p2", pinned=[case])

    assert case.id not in [item.chunk_id for item in plain.evidence.items]
    assert pinned.evidence.items[0].chunk_id == case.id and pinned.evidence.items[0].source_id == "S1"
    assert len(pinned.evidence.items) == len(plain.evidence.items) == settings.retrieval.k_final
    ids = [item.chunk_id for item in pinned.evidence.items]
    assert len(ids) == len(set(ids))


def test_a_chosen_passage_that_retrieval_also_found_is_listed_once(retriever):
    settings, r = retriever
    question = "임차권등기명령 신청서에 적어야 하는 사항은?"
    found = r.retrieve(question, settings.retrieval.k_final, None)[1].chunk
    plain = LegalAnswer(summary="일반적인 안내입니다.", analysis="일반적인 절차만 설명합니다.", confidence=Confidence.low)
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis(issues=["보증금 반환"])], "LegalAnswer": [plain] * 3})

    response = _orchestrator(settings, r, model).handle(question, conversation_id="p3", pinned=[found])

    ids = [item.chunk_id for item in response.evidence.items]
    assert ids[0] == found.id and ids.count(found.id) == 1


def test_the_draft_streams_each_attempt_and_the_answer_is_unchanged(retriever):
    settings, r = retriever
    bad = _answer(source_id="S9", reference="민법 제839조의2", analysis="민법 제839조의2에 따라 청구할 수 있습니다. 보증금은 반환되어야 합니다.")
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis(issues=["보증금 반환"])], "LegalAnswer": [bad, _answer()]})
    events = []
    resp = _orchestrator(settings, r, model).handle("보증금을 못 받았어요", conversation_id="d1", on_draft=events.append)
    assert resp.kind == "answer" and resp.trace.regenerations == 1
    # Only the analysis streams, and the regeneration throws the first draft away.
    assert events == [None, bad.analysis, None, resp.answer.analysis]


def test_a_clarification_streams_no_draft(retriever):
    settings, r = retriever
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis(needs_more_info=True, missing_facts=["보증금 액수"])]})
    events = []
    assert _orchestrator(settings, r, model).handle("어떻게 되나요?", conversation_id="d2", on_draft=events.append).kind == "clarification"
    assert events == []


def test_each_stage_is_announced_as_it_starts(retriever):
    settings, r = retriever
    bad = _answer(source_id="S9", reference="민법 제839조의2", analysis="민법 제839조의2에 따라 청구할 수 있습니다. 보증금은 반환되어야 합니다.")
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis(issues=["보증금 반환"])], "LegalAnswer": [bad, _answer()]})
    stages = []
    _orchestrator(settings, r, model).handle("보증금을 못 받았어요", conversation_id="s1", on_stage=stages.append)
    assert stages == ["analyze", "search", "write", "verify", "rewrite", "verify"]


def test_a_clarification_announces_only_the_analysis(retriever):
    settings, r = retriever
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis(needs_more_info=True, missing_facts=["보증금 액수"])]})
    stages = []
    _orchestrator(settings, r, model).handle("어떻게 되나요?", conversation_id="s2", on_stage=stages.append)
    assert stages == ["analyze"]

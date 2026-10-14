"""Frozen turns: retrieval and analysis are recorded once, then replayed so only generation varies."""
from legal_rag.evaluation.answers.frozen import FrozenAnalyzer, FrozenRetriever, capture_turn, load_frozen, save_frozen
from legal_rag.llm import FakeChatModel
from legal_rag.rag.conversation import InMemoryConversationStore
from legal_rag.rag.orchestrator import Orchestrator
from legal_rag.schemas import Citation, CitationKind, Confidence, LegalAnswer, QueryAnalysis
from legal_rag.verification import VerificationPolicy, Verifier

QUESTION = "임차권등기명령 신청서에 적어야 하는 사항은?"


def _answer():
    return LegalAnswer(summary="신청할 수 있습니다.", analysis="주택임대차보호법 제3조의3에 따라 신청할 수 있습니다.",
                       citations=[Citation(source_id="S1", kind=CitationKind.statute, reference="주택임대차보호법 제3조의3")],
                       confidence=Confidence.medium)


def _orchestrator(settings, retriever, model):
    return Orchestrator.from_components(settings=settings, retriever=retriever, model=model, store=InMemoryConversationStore(),
                                        verifier=Verifier(), policy=VerificationPolicy(max_regenerations=settings.rag.max_regenerations))


def test_capture_records_analysis_and_hits_without_generating(indexed):
    settings, _, hybrid = indexed
    analysis = QueryAnalysis(domain="민사", needs_more_info=True, missing_facts=["보증금 액수"])
    model = FakeChatModel({"QueryAnalysis": [analysis]})  # no LegalAnswer scripted: generation must not be called
    turn = capture_turn(_orchestrator(settings, hybrid, model), QUESTION)
    assert turn["question"] == QUESTION and turn["analysis"]["missing_facts"] == ["보증금 액수"]
    assert turn["hits"] and turn["hits"][0]["chunk"]["text"]
    assert [call.response_schema.__name__ for call in model.calls] == ["QueryAnalysis"]


def test_a_frozen_turn_replays_the_same_evidence_and_open_questions(indexed, tmp_path):
    settings, _, hybrid = indexed
    analysis = QueryAnalysis(domain="민사", needs_more_info=True, missing_facts=["보증금 액수"])
    live = _orchestrator(settings, hybrid, FakeChatModel({"QueryAnalysis": [analysis], "LegalAnswer": [_answer()]}))
    expected = live.handle(QUESTION, conversation_id="live").evidence

    turn = capture_turn(_orchestrator(settings, hybrid, FakeChatModel({"QueryAnalysis": [analysis]})), QUESTION)
    save_frozen(tmp_path / "frozen.jsonl", [{"key": "k1", **turn}])
    frozen = load_frozen(tmp_path / "frozen.jsonl")["k1"]

    model = FakeChatModel({"LegalAnswer": [_answer()]})  # no QueryAnalysis scripted: the analyzer must not call the model
    orch = _orchestrator(settings, FrozenRetriever(), model)
    orch.analyzer = FrozenAnalyzer()
    orch.retriever.load(frozen)
    orch.analyzer.load(frozen)
    response = orch.handle(QUESTION, conversation_id="replay")

    assert response.kind == "answer" and response.evidence == expected
    assert "보증금 액수" in " ".join(m.content for m in model.calls[0].messages)


def test_a_frozen_retriever_refuses_a_question_it_was_not_loaded_for():
    import pytest

    retriever = FrozenRetriever()
    retriever.load({"question": QUESTION, "analysis": {}, "hits": []})
    with pytest.raises(ValueError):
        retriever.retrieve("다른 질문", 5, None)

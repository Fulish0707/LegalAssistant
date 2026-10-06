from datetime import datetime

from legal_rag.llm import FakeChatModel
from legal_rag.rag.condenser import QueryCondenser
from legal_rag.schemas import StandaloneQuestion, Turn


def _turns(*pairs):
    return [Turn(role=role, content=text, created_at=datetime(2026, 1, 1)) for role, text in pairs]


HISTORY = _turns(("user", "해고당한 지 두 달 됐는데 부당해고 구제신청을 할 수 있나요?"), ("assistant", "3개월 이내라면 노동위원회에 신청할 수 있습니다."))


def test_without_history_the_message_is_returned_and_the_model_is_not_called():
    model = FakeChatModel({})
    assert QueryCondenser(model).condense("그 신청은 어디에 하나요?", []) == "그 신청은 어디에 하나요?"
    assert model.calls == []


def test_a_follow_up_becomes_a_standalone_question():
    model = FakeChatModel({"StandaloneQuestion": [StandaloneQuestion(question="부당해고 구제신청은 어디에 하나요?")]})
    assert QueryCondenser(model).condense("그 신청은 어디에 하나요?", HISTORY) == "부당해고 구제신청은 어디에 하나요?"
    call = model.calls[0]
    assert call.temperature == 0.0 and call.messages[0].role == "system"
    content = call.messages[-1].content
    assert "부당해고 구제신청을 할 수 있나요" in content and "노동위원회" in content and "현재 질문: 그 신청은 어디에 하나요?" in content


def test_only_recent_turns_and_a_bounded_slice_of_each_are_sent():
    history = _turns(*[("user" if i % 2 == 0 else "assistant", f"turn {i} " + "가" * 900) for i in range(10)])
    model = FakeChatModel({"StandaloneQuestion": [StandaloneQuestion(question="q")]})
    QueryCondenser(model, history_turns=2).condense("질문", history)
    content = model.calls[0].messages[-1].content
    assert "turn 9" in content and "turn 8" in content and "turn 7" not in content
    assert "가" * 900 not in content


def test_a_model_failure_falls_back_to_the_message():
    class Broken:
        name = "broken"

        def generate(self, *args, **kwargs):
            raise RuntimeError("down")

    assert QueryCondenser(Broken()).condense("그 신청은 어디에 하나요?", HISTORY) == "그 신청은 어디에 하나요?"


def test_an_empty_or_runaway_answer_falls_back_to_the_message():
    for bad in ("   ", "가" * 1500):
        model = FakeChatModel({"StandaloneQuestion": [StandaloneQuestion(question=bad)]})
        assert QueryCondenser(model).condense("그 신청은 어디에 하나요?", HISTORY) == "그 신청은 어디에 하나요?"

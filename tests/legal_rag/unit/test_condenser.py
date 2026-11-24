from datetime import datetime

from legal_rag.llm import FakeChatModel
from legal_rag.rag.condenser import QueryCondenser, is_standalone_question
from legal_rag.rag.conversation import render_turns
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
        assert QueryCondenser(model).condense("그 신청은 어디에 하나요?", []) == "그 신청은 어디에 하나요?"
        # with history the fallback joins the user's words; the bad output never becomes the query
        assert QueryCondenser(model).condense("그 신청은 어디에 하나요?", HISTORY) == (
            "해고당한 지 두 달 됐는데 부당해고 구제신청을 할 수 있나요? 그 신청은 어디에 하나요?")


# Condensed questions recorded by the multi-turn diagnostic (data/eval/multiturn_check, run
# 20261124T091824). Development cases for the form check: they shaped it, they do not prove it.
DIAGNOSTIC_QUESTIONS = [
    "전세 계약이 종료되어 이사를 가야 하는 상황에서, 집주인이 보증금을 돌려주지 않고 있습니다. 이사를 진행하면서도 보증금 반환을 요구하고 권리를 보호하기 위한 구체적인 절차는 무엇인가요?",
    "퇴사 후 마지막 달 월급과 퇴직금을 지연받은 상황에서, 근로기준법에 따라 지연이자를 청구할 수 있는 조건은 무엇인가요?",
    "신호 대기 중 뒤차와의 충돌로 인해 목 부상을 입었고, 보험사가 제시한 합의금이 적절하지 않다고 느껴질 때, 어떻게 대응해야 하나요?",
    "결혼 전에 구입한 주택에 대한 재산분할 청구를 이혼 후 언제까지 할 수 있나요?",
    "중고거래로 돈을 보냈으나 물건을 받지 못한 상황에서 경찰에 신고했을 때 돈을 돌려받을 수 있는 가능성은 무엇인가요?",
    "집주인이 보일러 고장 수리를 거부하거나 지연시키는 경우, 민법 제626조에 따라 임차인은 수리 비용을 지출한 후 임대인에게 그 비용의 상환을 청구할 수 있나요?",
    "친구에게 500만 원을 빌려준 지 1년이 지났지만 갚지 않고 있습니다. 이 상황에서 법적으로 취할 수 있는 조치는 무엇인가요?",
    "회사가 연차 휴가 사용을 제한하고 있는데, 이는 법적으로 문제가 없는 상황인가요?",
    "근로기준법 제23조를 위반하여 근로자에게 부당한 해고, 휴직, 정직, 전직, 감봉 등의 징벌을 가한 경우, 사용자는 법적인 제재를 받을 수 있으며 구체적으로는 어떤 처벌이나 조치가 적용되는지 알려주세요?",
]
DIAGNOSTIC_NOT_QUESTIONS = [
    # M02: an answer, three sentences, though it ends with a question
    "회사로부터 갑작스러운 해고 통보를 받은 경우, 구제신청은 구체적인 해고 절차와 규정에 따라 다르지만, 일반적으로 해고 통지를 받은 날로부터 일정 기간 내에 신청해야 합니다. 정확한 기간은 근로기준법에 따라 달라질 수 있으니, 해당 법을 확인하거나 회사의 내부 규정을 확인해 보시는 것이 좋습니다. 구체적인 기간을 알고 싶으시다면, 언제까지 신청해야 하는지 명시된 규정이나 법률 조항을 알려주시겠습니까?",
    # M06: a source handle copied from the previous answer
    "아버지의 빚 규모와 채무 내역이 불분명한 상황에서, 2026년 3월 17일 시행 예정인 [LEGAL SOURCE S4]의 조항이 적용되지 않을 경우, 상속인들은 현재로서는 어떤 법적 조치를 취해야 할까요?",
    # M07: an answer
    "거짓말 내용이 구체적으로 무엇인지 알려주신다면, 고소를 위해 필요한 증거 자료와 절차는 다음과 같습니다: 거짓말 게시 내용의 스크린샷이나 원본 링크, 거짓말 게시자의 신원 확인 방법 및 증거 (예: IP 주소 추적 결과, 계정 정보 등), 그리고 이러한 행위로 인한 구체적인 손해 내용을 준비해야 합니다.",
    # M09 third turn: a judgement, not a question
    "보일러 수리 비용을 직접 지출하고 그 금액을 월세에서 차감해도 되는지, 임대차 계약서에 명시된 유지보수 조항과 집주인의 동의 여부에 따라 달라질 수 있습니다.",
]


def test_the_form_check_on_the_diagnostic_outputs():
    assert [q for q in DIAGNOSTIC_QUESTIONS if not is_standalone_question(q)] == []
    assert [q for q in DIAGNOSTIC_NOT_QUESTIONS if is_standalone_question(q)] == []


def test_each_rule_of_the_form_check_rejects_on_its_own():
    assert is_standalone_question("부당해고 구제신청은 어디에 하나요?")
    assert is_standalone_question("근로기준법 제23조가 무슨 내용인지 알려주세요.")
    assert is_standalone_question("보증금을 돌려받지 못했습니다. 어떻게 해야 하나요")  # no mark, interrogative ending
    assert not is_standalone_question("부당해고 구제신청은 [S2]에 따라 어디에 하나요?")  # handle only
    assert not is_standalone_question("부당해고 구제신청은 [S1, S3]에 따라 어디에 하나요?")
    assert not is_standalone_question("해고됐습니다. 두 달 지났습니다. 구제신청은 어디에 하나요?")  # three sentences only
    assert not is_standalone_question("구제신청은 노동위원회에 하면 됩니다.")  # declarative ending only
    assert not is_standalone_question("")
    assert not is_standalone_question("   ")


def test_a_number_with_a_decimal_point_is_not_a_sentence_break():
    assert is_standalone_question("지연이자 연 20% 와 연 1.5% 중 무엇이 적용되나요?")


def test_an_answer_instead_of_a_question_falls_back_to_the_users_words():
    history = _turns(("user", "회사에서 갑자기 다음 주부터 나오지 말라고 했어요."), ("assistant", "해고 통보로 볼 수 있습니다."))
    model = FakeChatModel({"StandaloneQuestion": [StandaloneQuestion(question=DIAGNOSTIC_NOT_QUESTIONS[0])]})
    assert QueryCondenser(model).condense("구제신청은 언제까지 해야 하나요?", history) == (
        "회사에서 갑자기 다음 주부터 나오지 말라고 했어요. 구제신청은 언제까지 해야 하나요?")


def test_the_fallback_uses_only_user_turns_inside_the_window():
    history = _turns(("user", "오래된 질문"), ("assistant", "답"), ("user", "최근 질문"), ("assistant", "답2"))
    model = FakeChatModel({"StandaloneQuestion": [StandaloneQuestion(question="답변입니다.")]})
    assert QueryCondenser(model, history_turns=2).condense("현재 질문", history) == "최근 질문 현재 질문"


def test_source_handles_from_earlier_answers_do_not_reach_the_condenser():
    history = _turns(("user", "아버지가 빚을 남기고 돌아가셨어요."),
                     ("assistant", "상속 포기는 [LEGAL SOURCE S4]에 따릅니다 [S4]. 한정승인도 있습니다 [S1, S2]."))
    model = FakeChatModel({"StandaloneQuestion": [StandaloneQuestion(question="상속 포기 기한이 지나면 어떻게 되나요?")]})
    QueryCondenser(model).condense("그 기한이 지나면 어떻게 되나요?", history)
    content = model.calls[0].messages[-1].content
    assert "LEGAL SOURCE" not in content and "[S" not in content
    assert "한정승인도 있습니다" in content


def test_render_turns_drops_source_handles_for_every_reader():
    # The generator reads the same lines; an old S4 is not the new bundle's S4.
    lines = render_turns(_turns(("assistant", "기한은 3개월입니다[S2]. [LEGAL SOURCE S4] 참조, [S1, S3]과 [S10].")))
    assert lines == ["- 상담: 기한은 3개월입니다. 참조, 과 ."]
    assert render_turns(_turns(("user", "[S2] 문서를 봤어요"))) == ["- 사용자: [S2] 문서를 봤어요"]  # the user's words stay

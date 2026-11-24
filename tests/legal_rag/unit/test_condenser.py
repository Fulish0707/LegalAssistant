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


def _condense(question, message, history, **kwargs):
    model = FakeChatModel({"StandaloneQuestion": [StandaloneQuestion(question=question)]})
    return QueryCondenser(model, **kwargs).condense(message, history)


# Outputs recorded by the multi-turn runs (data/eval/condense_facts/dev_scan.py): development cases
# for the fact check. They shaped it; they do not prove it.
def test_invented_quantities_from_the_recorded_runs_are_not_used():
    cases = [
        # base N09: the date is the condense prompt's own example
        (_turns(("user", "전세로 사는 집이 경매에 넘어갔어요."), ("assistant", "경매 절차에서 임차인은 배당을 요구할 수 있습니다.")),
         "보증금은 어떻게 되나요?",
         "전세 계약 기간이 끝나기 전에 집이 경매로 넘어갔으며, 9월 1일에 보증금 규모는 얼마였는지, 현재 경매 진행 상황에서 보증금 반환 가능성과 절차는 어떻게 되는지 알려주세요."),
        # fix N07
        (_turns(("user", "친구한테 빌려준 돈을 받으려고 해요. 차용증은 없어요."), ("assistant", "차용증이 없어도 송금 기록으로 입증할 수 있습니다.")),
         "그 돈을 청구할 수 있는 기한이 있나요?",
         "친구에게 빌려준 돈의 경우, 구체적으로 2023년 3월에 500만원을 빌려준 상황이라면, 이 돈을 청구할 수 있는 법적 기한은 어떻게 되나요?"),
        # fix M03: dates and periods, one of them (14일) from the previous answer
        (_turns(("user", "퇴사했는데 마지막 달 월급을 못 받았어요."), ("assistant", "퇴직 후 14일 이내에 임금을 지급해야 합니다.")),
         "지연이자도 받을 수 있나요?",
         "퇴사일이 2023년 3월 15일이고 마지막 달 근무 일수는 20일이었는데, 퇴직 후 14일 이내에 임금 및 퇴직금을 지급받지 못해 지연이자를 포함한 금액을 청구할 수 있나요?"),
        # diagnostic M06: a date from the previous answer
        (_turns(("user", "아버지가 빚을 남기고 돌아가셨어요."), ("assistant", "2026년 3월 17일 시행 예정인 조항이 있습니다.")),
         "그 기한이 지나면 어떻게 되나요?",
         "아버지의 빚 규모와 채무 내역이 불분명한 상황에서, 2026년 3월 17일 시행 예정인 조항이 적용되지 않을 경우, 상속인들은 어떤 법적 조치를 취해야 할까요?"),
    ]
    for history, message, invented in cases:
        user = next(turn.content for turn in history if turn.role == "user")
        assert _condense(invented, message, history) == f"{user} {message}"


def test_quantities_the_user_gave_pass_however_they_are_written():
    # diagnostic M10: the user's own amount and period restated
    history = _turns(("user", "법적으로 어떻게 되나요?"), ("assistant", "어떤 일이 있었는지 알려 주세요."))
    restated = "친구에게 500만 원을 빌려준 지 1년이 지났지만 갚지 않고 있습니다. 이 상황에서 법적으로 취할 수 있는 조치는 무엇인가요?"
    assert _condense(restated, "친구에게 500만원을 빌려줬는데 1년째 안 갚아요.", history) == restated
    # spelled numbers become digits
    history = _turns(("user", "해고당한 지 두 달 됐어요."), ("assistant", "구제신청을 검토할 수 있습니다."))
    assert _condense("해고된 지 2개월이 지났는데 구제신청은 어디에 하나요?", "그건 어디에 하나요?", history).startswith("해고된 지 2개월")


def test_an_ordinal_looking_duration_of_the_user_counts():
    # the extractor drops "석 달째" as an ordinal; on the user's side it is a duration
    history = _turns(("user", "월세 50만 원을 석 달째 못 냈어요."), ("assistant", "차임 연체는 계약 해지 사유가 될 수 있습니다."))
    question = "월세 50만 원을 3개월 동안 못 냈는데 집주인이 바로 나가라고 할 수 있나요?"
    assert _condense(question, "집주인이 바로 나가라고 할 수 있나요?", history) == question


def test_a_date_shortened_from_the_users_date_passes():
    history = _turns(("user", "2025년 10월 1일에 교통사고가 났어요."), ("assistant", "손해배상을 청구할 수 있습니다."))
    question = "10월 1일에 난 교통사고의 손해배상 청구 기한은 언제까지인가요?"
    assert _condense(question, "기한이 있나요?", history) == question


def test_a_quantity_from_outside_the_window_is_invented():
    history = _turns(("user", "보증금 1억 원이에요."), ("assistant", "확인했습니다."), ("user", "집주인이 연락이 안 돼요."),
                     ("assistant", "내용증명을 보낼 수 있습니다."))
    # history_turns=2 keeps only the last two turns, so the user's 1억 is outside the window
    assert _condense("보증금 1억 원을 돌려받으려면 어떻게 하나요?", "어떻게 하나요?", history, history_turns=2) == (
        "집주인이 연락이 안 돼요. 어떻게 하나요?")


def test_articles_and_case_numbers_must_be_in_the_conversation():
    history = _turns(("user", "근로기준법 제23조가 무슨 내용인가요?"), ("assistant", "제23조는 해고를 제한하고 제28조는 구제신청을 정합니다."))
    ok = "근로기준법 제28조에 따른 구제신청은 어디에 하나요?"  # the article is in the previous answer
    assert _condense(ok, "그 신청은 어디에 하나요?", history) == ok
    invented_article = "근로기준법 제109조에 따른 처벌은 무엇인가요?"
    assert _condense(invented_article, "처벌은요?", history) == "근로기준법 제23조가 무슨 내용인가요? 처벌은요?"
    invented_case = "대법원 2019다12345 판결처럼 해고가 무효가 되나요?"
    assert _condense(invented_case, "무효가 되나요?", history) == "근로기준법 제23조가 무슨 내용인가요? 무효가 되나요?"


def test_a_question_without_quantities_or_references_is_used():
    assert _condense("부당해고 구제신청은 어디에 하나요?", "그 신청은 어디에 하나요?", HISTORY) == "부당해고 구제신청은 어디에 하나요?"

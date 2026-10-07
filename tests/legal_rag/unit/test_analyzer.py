from datetime import datetime

from legal_rag.llm import FakeChatModel
from legal_rag.rag.analyzer import QueryAnalyzer
from legal_rag.rag.clarify import ClarificationPolicy
from legal_rag.schemas import QueryAnalysis, Turn


def test_analyzer_merges_deterministic_refs_over_model_output():
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis(domain="민사", subdomain="임대차", issues=["보증금 반환"], keywords=["보증금"], statute_refs=["엉뚱한 법 제1조"])]})
    analysis = QueryAnalyzer(model).analyze("주택임대차보호법 제3조의3에 따라 집주인이 보증금을 안 줍니다", history=[])
    assert analysis.statute_refs == ["주택임대차보호법 제3조의3"] and analysis.domain == "민사"
    sent = model.calls[0].messages
    assert sent[0].role == "system" and "집주인" in sent[-1].content and model.calls[0].temperature == 0.0


def test_analyzer_includes_recent_history_only():
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis()]})
    history = [Turn(role="user" if i % 2 == 0 else "assistant", content=f"turn {i}", created_at=datetime(2026, 1, 1)) for i in range(10)]
    QueryAnalyzer(model, history_turns=2).analyze("질문", history=history)
    content = model.calls[0].messages[-1].content
    assert "turn 9" in content and "turn 8" in content and "turn 7" not in content


def test_analyzer_falls_back_to_deterministic_analysis_on_model_failure():
    class Broken:
        name = "broken"

        def generate(self, *args, **kwargs):
            raise RuntimeError("down")

    analysis = QueryAnalyzer(Broken()).analyze("민법 제750조 손해배상 2018다12345", history=[])
    assert analysis.statute_refs == ["민법 제750조"] and analysis.case_refs == ["2018다12345"] and analysis.needs_more_info is False


def test_clarification_policy():
    p = ClarificationPolicy(allow=True)
    assert p.should_clarify(QueryAnalysis(needs_more_info=True, missing_facts=["계약 종료일"])) is True
    assert p.should_clarify(QueryAnalysis(needs_more_info=True, missing_facts=["x"], statute_refs=["민법 제750조"])) is False
    assert p.should_clarify(QueryAnalysis(needs_more_info=True)) is False
    assert ClarificationPolicy(allow=False).should_clarify(QueryAnalysis(needs_more_info=True, missing_facts=["x"])) is False


def test_a_question_with_a_legal_area_is_answered_not_questioned():
    # Asking instead of answering stalled conversations: 8 of 10 replies to a clarification were
    # questioned again (2026-10-05). With an area to search, the answer carries the open questions.
    p = ClarificationPolicy(allow=True)
    assert p.should_clarify(QueryAnalysis(domain="형사", needs_more_info=True, missing_facts=["혈중알코올농도"])) is False
    assert p.should_clarify(QueryAnalysis(domain=None, needs_more_info=True, missing_facts=["어떤 일이 있었나요?"])) is True


def _turns(*pairs):
    return [Turn(role=role, content=content, created_at=datetime(2026, 1, 1)) for role, content in pairs]


def test_a_reference_from_the_assistants_earlier_answer_is_not_the_users():
    """The model copied 근로기준법 제28조 and a case number out of the previous answer on every follow-up
    turn (2026-10-06). As "the user's references" they became search queries and were exempt from
    the citation check."""
    history = _turns(("user", "회사에서 해고 통보를 받았습니다."), ("assistant", "근로기준법 제28조에 따라 구제를 신청할 수 있습니다(93헌마13)."))
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis(statute_refs=["근로기준법 제28조 제2항", "대구지방법원의 판결(LEGAL SOURCE S4)"], case_refs=["93헌마13"])]})
    analysis = QueryAnalyzer(model).analyze("그건 어디에 신청하나요?", history=history)
    assert analysis.statute_refs == [] and analysis.case_refs == []


def test_a_reference_the_user_wrote_in_an_earlier_turn_is_still_theirs():
    history = _turns(("user", "민법 제618조와 2019다12345 판결이 제 경우에 적용되나요?"), ("assistant", "주택임대차보호법 제3조도 살펴봐야 합니다."))
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis(statute_refs=["민법 제618조", "주택임대차보호법 제3조"], case_refs=["2019다12345"])]})
    analysis = QueryAnalyzer(model).analyze("그럼 언제까지 해야 하나요?", history=history)
    assert analysis.statute_refs == ["민법 제618조"] and analysis.case_refs == ["2019다12345"]


def test_a_law_the_user_named_without_an_article_is_theirs():
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis(statute_refs=["계량에 관한 법률", "민법"])]})
    analysis = QueryAnalyzer(model).analyze("계량에 관한 법률에 따른 법정단위는 어떻게 구분되나요?", history=[])
    assert analysis.statute_refs == ["계량에 관한 법률"]

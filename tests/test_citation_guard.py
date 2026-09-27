"""Flag statute articles and case numbers the answer cites but no evidence contains.

With the legal search down, qwen3:4b answered a 전세 보증금 question with
"민법 제543조, 제544조" (2026-09-27), which is invented. The prompt already forbids that;
this is the check that does not depend on the model obeying.
"""

import unittest

from citation_guard import annotate_unsupported, unsupported_citations


class UnsupportedCitationTests(unittest.TestCase):
    def test_article_missing_from_evidence_is_flagged(self):
        answer = "법적 근거: 민법 제543조(보증금), 민법 제544조"
        self.assertEqual(unsupported_citations(answer, ["임대차 종료 후 보증금 반환"]), ["민법 제543조", "민법 제544조"])

    def test_article_present_in_evidence_passes_despite_spacing(self):
        answer = "근로기준법 제28조 제2항에 따라 3개월 이내입니다."
        evidence = ["헌법재판소 2011헌마233 근로기준법 제 28 조 제2항 등 위헌확인"]
        self.assertEqual(unsupported_citations(answer, evidence), [])

    def test_same_article_number_under_another_law_is_flagged(self):
        self.assertEqual(unsupported_citations("민법 제28조를 보면", ["근로기준법 제28조"]), ["민법 제28조"])

    def test_bare_article_passes_when_the_number_is_in_evidence(self):
        self.assertEqual(unsupported_citations("제3조의3에 따라", ["주택임대차보호법 제3조의3(임차권등기명령)"]), [])

    def test_case_numbers_are_checked_too(self):
        answer = "대법원 2010다95062 판결과 2099다1 판결"
        self.assertEqual(unsupported_citations(answer, ["… 대법원 2010다95062 참조"]), ["2099다1"])

    def test_the_users_own_text_counts_as_evidence(self):
        self.assertEqual(unsupported_citations("말씀하신 민법 제750조는", ["민법 제750조가 적용되나요?"]), [])

    def test_annotation_is_appended_once_and_only_when_needed(self):
        self.assertEqual(annotate_unsupported("답변", []), "답변")
        annotated = annotate_unsupported("답변", ["민법 제543조"])
        self.assertTrue(annotated.startswith("답변\n\n"))
        self.assertIn("민법 제543조", annotated)
        self.assertIn("확인", annotated)


if __name__ == "__main__":
    unittest.main()


import json as _json
from langchain_core.messages import AIMessage
from tests.test_chat_streaming import ChatStreamBase, StubAgent


def _schema(chat):
    return _json.dumps({"chat_response": chat, "statement_suggestion": "", "job_statement_suggestion": "",
                        "report_suggestion": ""}, ensure_ascii=False)


class CitationGuardStreamTests(ChatStreamBase):
    def final_for(self, chat):
        agent = StubAgent([("values", {"messages": [AIMessage(content=_schema(chat))]})])
        return [e for e in self.collect(agent, "message=보증금 문의") if e["type"] == "final_response"][0]

    def test_unsupported_article_in_the_answer_gets_a_notice(self):
        final = self.final_for("법적 근거: 민법 제543조")
        self.assertTrue(final["chat_response"].startswith("법적 근거: 민법 제543조\n\n"))
        self.assertIn("확인", final["chat_response"].split("\n\n", 1)[1])

    def test_answer_without_citations_is_untouched(self):
        self.assertEqual(self.final_for("보증금은 돌려받을 수 있습니다.")["chat_response"], "보증금은 돌려받을 수 있습니다.")

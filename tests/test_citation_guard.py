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

    def test_former_law_wording_is_not_a_law_name(self):
        """"구법 제62조" refers to an older version of the law under discussion; 구법 is no title."""
        answer = "같은 법 제71조 제11호(또는 구법 제62조)에서 처벌합니다."
        evidence = ["정보통신망법 제71조 제11호", "구 정보통신망법 제62조 제4호"]
        self.assertEqual(unsupported_citations(answer, evidence), [])

    def test_former_law_article_missing_from_evidence_is_still_flagged(self):
        self.assertEqual(unsupported_citations("구법 제62조에 따릅니다.", ["제71조"]), ["제62조"])

    def test_bare_article_passes_when_the_number_is_in_evidence(self):
        self.assertEqual(unsupported_citations("제3조의3에 따라", ["주택임대차보호법 제3조의3(임차권등기명령)"]), [])

    def test_branch_article_does_not_stand_in_for_the_main_article(self):
        self.assertEqual(unsupported_citations("주택임대차보호법 제3조", ["주택임대차보호법 제3조의3(임차권등기명령)"]),
                         ["주택임대차보호법 제3조"])

    def test_possessive_after_the_article_is_not_a_branch_number(self):
        """"「지방자치법」 제124조의 규정을 준용한다" in the evidence did not count as 제124조, so a
        verified answer was flagged (2026-09-28, 80-question rerun)."""
        evidence = ["교육감의 권한대행에 관하여는 「지방자치법」 제124조의 규정을 준용한다."]
        self.assertEqual(unsupported_citations("지방자치법 제124조에 따라", evidence), [])

    def test_case_numbers_are_checked_too(self):
        answer = "대법원 2010다95062 판결과 2099다1 판결"
        self.assertEqual(unsupported_citations(answer, ["… 대법원 2010다95062 참조"]), ["2099다1"])

    def test_generic_words_are_not_law_names(self):
        """"관련 법령 제48조" was read as a law named 법령 and flagged although 제48조 was in evidence."""
        self.assertEqual(unsupported_citations("장관의 권한 관련 법령 제48조에 따라", ["… 제48조(권한의 위임) …"]), [])
        self.assertEqual(unsupported_citations("같은 법률 제9조", ["없음"]), ["제9조"])

    def test_short_and_full_law_names_match_each_other(self):
        """"여성기업지원법 제2조" was flagged against "여성기업지원에 관한 법률 제2조" (2026-09-27)."""
        full = ["여성기업지원에 관한 법률 제2조(정의) 이 법에서 사용하는 용어의 뜻은"]
        self.assertEqual(unsupported_citations("여성기업지원법 제2조에 따르면", full), [])
        self.assertEqual(unsupported_citations("여성기업지원에 관한 법률 제2조", ["여성기업지원법 제2조"]), [])

    def test_short_name_still_needs_the_same_law_and_article(self):
        full = ["여성기업지원에 관한 법률 제2조(정의)"]
        self.assertEqual(unsupported_citations("근로기준법 제2조", full), ["근로기준법 제2조"])
        self.assertEqual(unsupported_citations("여성기업지원법 제9조", full), ["여성기업지원법 제9조"])

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


from tests.test_chat_streaming import ChatStreamBase, backend_answer


class CitationGuardStreamTests(ChatStreamBase):
    def final_for(self, chat):
        return self.final(self.collect(backend_answer(chat)))

    def test_unsupported_article_in_the_answer_gets_a_notice(self):
        final = self.final_for("법적 근거: 민법 제543조")
        self.assertTrue(final["chat_response"].startswith("법적 근거: 민법 제543조\n\n"))
        self.assertIn("확인", final["chat_response"].split("\n\n", 1)[1])

    def test_answer_without_citations_is_untouched(self):
        self.assertEqual(self.final_for("보증금은 돌려받을 수 있습니다.")["chat_response"], "보증금은 돌려받을 수 있습니다.")

    def test_an_article_deep_in_the_evidence_is_not_flagged(self):
        """The source cards show a 400-character excerpt; the guard must read the whole block."""
        item = {"source_id": "S1", "document_type": "judgment", "score": 0.9, "header": {"Type": "Case"},
                "text": "가" * 600 + " 민법 제543조에 따라 해지할 수 있다."}
        final = self.final(self.collect(backend_answer("민법 제543조에 따라 해지할 수 있습니다.", items=[item])))

        self.assertNotIn("확인 필요", final["chat_response"])

    def test_what_the_last_answer_cited_is_not_flagged_when_repeated(self):
        """The service lets a follow-up repeat the last answer's statute; the page must agree with it."""
        body = backend_answer("근로기준법 제36조에 따라 14일 이내에 지급해야 합니다.")
        self.assertIn("확인 필요: 근로기준법 제36조", self.final(self.collect(body))["chat_response"])
        body["prior_refs"] = ["근로기준법 제36조"]
        self.assertNotIn("확인 필요", self.final(self.collect(body))["chat_response"])

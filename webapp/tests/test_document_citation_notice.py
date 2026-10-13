import unittest

from llm_ocr import flag_unsourced_citations

DOCUMENT = "근로계약서\n6. 퇴직: 위약금 500만원\n제7조(비밀유지) 근로기준법 제17조에 따라 서면으로 교부한다. 사건 2025차전10432"


class DocumentCitationNoticeTests(unittest.TestCase):
    def test_an_article_the_document_does_not_contain_is_named_in_a_notice(self):
        report = flag_unsourced_citations("<p>위약금 조항은 근로기준법 <strong>제124조</strong> 위반입니다.</p>", [DOCUMENT])
        self.assertIn("※ 확인 필요: 근로기준법 제124조", report)
        self.assertTrue(report.startswith("<p>위약금 조항은"))

    def test_what_the_document_itself_contains_is_not_flagged(self):
        report = "<p>제7조와 제6조를 보면 근로기준법 제17조, 사건 2025차전10432</p>"
        self.assertEqual(flag_unsourced_citations(report, [DOCUMENT]), report)

    def test_a_case_number_and_a_law_whose_name_starts_like_an_article_are_flagged(self):
        report = flag_unsourced_citations("<p>대법원 2019다12345, 제주특별법 제3조</p>", [DOCUMENT])
        self.assertIn("2019다12345", report)
        self.assertIn("제주특별법 제3조", report)


if __name__ == "__main__":
    unittest.main()

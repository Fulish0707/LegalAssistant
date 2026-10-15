"""A document longer than the model's window is analysed from its start, and the report says so.

Ollama drops the start of a prompt that does not fit, mode instructions included, and the model
then reports the dropped clauses as absent from the document (2026-10-15).
"""
import logging
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import llm_ocr
from llm_prompt import prompt


class LongDocumentTests(unittest.TestCase):
    def analyse(self, text):
        sent = []
        llm = SimpleNamespace(invoke=lambda messages: sent.append(messages[-1].content) or
                              SimpleNamespace(content="<div><h2>분석</h2></div>"))
        with patch.object(llm_ocr, "extract_text_from_document", return_value=({"text": text}, 200)):
            result, status = llm_ocr.analyze_document("a.pdf", "pdf", "", llm, prompt, logging.getLogger("t"))
        self.assertEqual(status, 200)
        return sent[0], result["report_html"]

    def test_a_long_document_is_cut_to_its_start_and_the_report_says_so(self):
        text = "제1조 계약금 7,345,000원\n" + "가" * 20000 + "\n마지막 조항"
        sent, report = self.analyse(text)
        self.assertIn("7,345,000원", sent)
        self.assertNotIn("마지막 조항", sent)
        self.assertIn("제공된 앞부분에서는 찾을 수 없으며", sent)
        self.assertLess(len(sent), llm_ocr.DOCUMENT_ANALYSIS_MAX_CHARS + 3000)
        self.assertIn(f"{llm_ocr.DOCUMENT_ANALYSIS_MAX_CHARS:,}자", report)
        self.assertIn(f"{len(text):,}자", report)

    def test_a_short_document_is_sent_whole_without_a_notice(self):
        sent, report = self.analyse("제1조 계약금 7,345,000원\n마지막 조항")
        self.assertIn("마지막 조항", sent)
        self.assertNotIn("제공된 앞부분에서는 찾을 수 없으며", sent)
        self.assertNotIn("문서가 길어", report)


if __name__ == "__main__":
    unittest.main()

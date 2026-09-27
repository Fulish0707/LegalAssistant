"""Behavioural contracts on the chat agent's retrieval rules in llm_prompt.

Live runs showed qwen3:4b answering legal-rule questions (deposit return case law,
unfair-dismissal filing deadline) without calling search_legal_corpus and inventing
article numbers and deadlines. These tests pin the prompt wording that forbids that.
"""

import re
import unittest

import llm_prompt


def _normalized(text):
    return re.sub(r"\s+", " ", text).strip().lower()


class ChatRetrievalPolicyTests(unittest.TestCase):
    def setUp(self):
        self.text = _normalized(llm_prompt.CHAT_OUTPUT_INSTRUCTIONS)

    def test_legal_rule_answers_require_search_legal_corpus_first(self):
        self.assertIn("search_legal_corpus", self.text)
        match = re.search(
            r"before (?:any )?answer[^.]*?(?:statute|article)[^.]*?deadline[^.]*?"
            r"requirement[^.]*?courts?[^.]*?search_legal_corpus",
            self.text,
        )
        self.assertIsNotNone(
            match,
            "Prompt must require calling search_legal_corpus before answers that "
            "state an article, deadline, legal requirement, or court ruling.",
        )

    def test_general_explanation_loophole_removed(self):
        self.assertNotIn("general explanations do not require retrieval", self.text)
        self.assertNotIn("general explanation", self.text)

    def test_never_cite_unretrieved_article_case_or_deadline(self):
        self.assertRegex(
            self.text,
            r"never (?:state|cite)[^.]*article number[^.]*case number[^.]*deadline"
            r"[^.]*(?:retrieved|search) results?[^.]*user",
        )

    def test_unsupported_numbers_replaced_by_needs_confirmation(self):
        self.assertRegex(self.text, r"(?:needs|requires) confirmation")
        self.assertIn("instead of", self.text)

    def test_greetings_clarifying_and_drafting_need_no_retrieval(self):
        self.assertRegex(
            self.text,
            r"greetings, clarifying questions,? and drafting from facts the user "
            r"supplied need no retrieval",
        )

    def test_search_documents_still_scoped_to_uploaded_documents(self):
        self.assertRegex(self.text, r"use search_documents[^.]*uploaded documents")

    def test_unavailable_tool_rule_preserved(self):
        self.assertIn("never claim live statute or case-law searches", self.text)
        self.assertIn("never pretend a tool was called", self.text)

    def test_korean_language_line_preserved(self):
        self.assertIn(
            "Write every field in Korean (한국어), even when retrieved sources, tool output or these\n"
            "instructions are in English, unless the user explicitly asked for another language.",
            llm_prompt.CHAT_OUTPUT_INSTRUCTIONS,
        )

    def test_output_schema_keys_preserved(self):
        for key in (
            "chat_response",
            "statement_suggestion",
            "job_statement_suggestion",
            "report_suggestion",
        ):
            self.assertIn(f"- {key}:", llm_prompt.CHAT_OUTPUT_INSTRUCTIONS)
        self.assertIn(
            "leave inapplicable suggestion fields as empty strings", self.text
        )


if __name__ == "__main__":
    unittest.main()

"""The user's uploaded case documents travel with a chat turn and come back as source cards."""

import copy
import json
import unittest
from unittest import mock

import httpx

import legal_backend
from tests.test_chat_streaming import ChatStreamBase, backend_answer
from tests.test_legal_backend import ANSWER, served

LEASE = {"name": "임대차계약서.pdf", "text": "제5조 임대인은 계약 종료일에 보증금 1억 원을 반환한다.", "score": 0.82}


class AskWithDocumentsTests(unittest.TestCase):
    def sent(self, *args):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=ANSWER)

        with served(handler):
            legal_backend.ask("http://legal:8000", "보증금 문의", "browser-1", *args)
        return seen

    def test_no_documents_sends_the_same_body_as_before(self):
        self.assertEqual(self.sent(), {"message": "보증금 문의", "conversation_id": "browser-1"})
        self.assertEqual(self.sent([]), {"message": "보증금 문의", "conversation_id": "browser-1"})

    def test_documents_are_sent_by_name_and_text_only(self):
        self.assertEqual(self.sent([LEASE])["user_documents"], [{"name": LEASE["name"], "text": LEASE["text"]}])


class RenderWithDocumentsTests(unittest.TestCase):
    def body(self, analysis):
        body = copy.deepcopy(ANSWER)
        body["answer"]["analysis"] = analysis
        return body

    def test_a_document_the_answer_uses_becomes_a_card_after_the_legal_sources(self):
        text, sources = legal_backend.render(self.body("계약서는 종료일에 반환한다고 정합니다(D1). 기간은 3개월입니다(S1)."), [LEASE])

        self.assertIn("정합니다[2]", text)
        self.assertIn("3개월입니다[1]", text)
        self.assertEqual(sources[-1], {"source": "임대차계약서.pdf", "quote": LEASE["text"], "kind": "내 문서", "score": 0.82})
        self.assertEqual(len(sources), 2)

    def test_a_document_the_answer_does_not_use_gets_no_card(self):
        text, sources = legal_backend.render(self.body("기간은 3개월입니다(S1)."), [LEASE])

        self.assertEqual([source["kind"] for source in sources], ["법령"])

    def test_the_models_long_form_of_a_document_id_is_numbered_too(self):
        text, _ = legal_backend.render(self.body("반환한다고 정합니다 [USER DOCUMENT D1]. 기간은 3개월입니다(S1)."), [LEASE])

        self.assertIn("정합니다 [2]", text)
        self.assertNotIn("USER DOCUMENT", text)


class ChatWithCaseDocumentsTests(ChatStreamBase):
    def collect_with(self, results, backend=None, ok=True):
        search = mock.Mock(return_value={"ok": ok, "results": results})
        self.search = search
        with mock.patch.object(self.app_module, "search_knowledge_base", search):
            return self.collect(backend)

    def stored(self, quote="제5조 임대인은 계약 종료일에 보증금 1억 원을 반환한다.", source="임대차계약서.pdf", score=0.82):
        return {"source": source, "score": score, "quote": quote, "analysis_mode": "report"}

    def test_this_browsers_documents_are_searched_with_the_message_and_sent(self):
        self.as_browser("browser-with-lease")
        self.collect_with([self.stored()])

        self.assertEqual(self.search.call_args.args[:2], ("browser-with-lease", "보증금 문의"))
        documents = self.ask.call_args.args[3]
        self.assertEqual([(d["name"], d["text"]) for d in documents], [("임대차계약서.pdf", self.stored()["quote"])])

    def test_a_browser_without_documents_calls_the_backend_as_before(self):
        self.collect_with([])

        self.assertEqual(self.ask.call_args.args[3], [])

    def test_a_long_passage_is_cut_to_what_the_backend_reads(self):
        self.collect_with([self.stored(quote="가" * 5000)])

        self.assertEqual(len(self.ask.call_args.args[3][0]["text"]), self.app_module.CASE_DOCUMENT_CHARS)

    def test_a_passage_far_from_the_message_is_not_sent(self):
        self.collect_with([self.stored(score=0.52), self.stored(source="무관한자료.pdf", score=0.12)])

        self.assertEqual([d["name"] for d in self.ask.call_args.args[3]], ["임대차계약서.pdf"])

    def test_a_failed_document_search_does_not_stop_the_answer(self):
        events = self.collect_with([], ok=False)

        self.assertEqual(self.ask.call_args.args[3], [])
        self.assertEqual(events[-2]["type"], "final_response")

    def test_a_used_document_is_shown_as_a_card(self):
        events = self.collect_with([self.stored()], backend_answer("계약서는 종료일에 반환한다고 정합니다(D1)."))

        self.assertEqual(self.final(events)["sources"], [{"source": "임대차계약서.pdf", "kind": "내 문서", "score": 0.82,
                                                         "quote": self.stored()["quote"]}])

    def test_an_article_written_in_the_document_gets_no_warning(self):
        answer = "계약서는 민법 제618조를 들어 반환을 정합니다(D1)."
        with_document = self.final(self.collect_with([self.stored(quote="제5조 민법 제618조에 따라 보증금을 반환한다.")], backend_answer(answer)))
        without = self.final(self.collect_with([], backend_answer(answer)))

        self.assertNotIn("확인 필요", with_document["chat_response"])
        self.assertIn("확인 필요: 민법 제618조", without["chat_response"])

    def test_a_contracts_own_clause_number_does_not_excuse_an_invented_article(self):
        """A lease numbers its clauses 제1조, 제2조 ...; that must not vouch for a law's 제4조."""
        lease = self.stored(quote="주택임대차보호법을 따른다. 제4조 (계약의 갱신) 2개월 전까지 알려야 한다.")
        final = self.final(self.collect_with([lease], backend_answer("주택임대차보호법 제4조에 따라 갱신을 거절할 수 있습니다(D1).")))

        self.assertIn("확인 필요: 주택임대차보호법 제4조", final["chat_response"])

    def test_pointing_to_the_contracts_own_clause_gets_no_warning(self):
        lease = self.stored(quote="제5조 (보증금의 반환) 갑은 종료일에 보증금 전액을 반환한다.")
        final = self.final(self.collect_with([lease], backend_answer("계약서 제5조에 따라 종료일에 반환해야 합니다(D1).")))

        self.assertNotIn("확인 필요", final["chat_response"])


if __name__ == "__main__":
    unittest.main()

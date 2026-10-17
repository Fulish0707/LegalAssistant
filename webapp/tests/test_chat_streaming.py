"""SSE contract tests for /chat.

The subject here is the event contract static/lc-chat.js consumes, not answer
quality: every test stubs legal_backend.ask, the call to legal_rag's /v1/chat/stream.
A real call would need the service, its indexes and a model.
"""

import importlib
import contextlib
import json
import os
import sys
import tempfile
import unittest
from urllib.parse import quote
from unittest import mock


# Exactly the set static/lc-chat.js branches on (sendMessage, sse.onmessage).
FRONTEND_EVENT_TYPES = {
    "thought",
    "draft",
    "draft_reset",
    "final_response",
    "error",
    "stream_end",
}


def turn(body):
    """What legal_backend.ask yields for a turn whose /v1/chat body is `body`, without draft events."""
    return [{"type": "response", "body": body}]


def backend_answer(summary="보증금은 돌려받을 수 있습니다.", analysis="", items=()):
    """A /v1/chat body of kind "answer"."""
    return {
        "kind": "answer",
        "answer": {"summary": summary, "analysis": analysis, "possible_actions": [],
                   "additional_information_needed": [], "uncertainty": "", "confidence": "medium"},
        "evidence": {"items": list(items)},
    }


STATUTE_ITEM = {"source_id": "S1", "chunk_id": "law#art:제3조", "document_type": "statute", "score": 0.9,
                "header": {"Type": "Statute", "Law": "주택임대차보호법", "Article": "제3조"},
                "text": "임차인은 보증금의 반환을 청구할 수 있다."}


class ChatStreamBase(unittest.TestCase):
    MODULE_NAMES = ("app", "langchain_config", "llm_model", "llm_prompt")

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.saved_env = {
            name: os.environ.get(name)
            for name in ("APP_STATE_DB", "RAG_PERSIST_DIR", "APP_ENV", "LANGSMITH_TRACING")
        }
        cls.saved_modules = {name: sys.modules.get(name) for name in cls.MODULE_NAMES}
        os.environ["APP_STATE_DB"] = os.path.join(cls.tmpdir.name, "app_state.db")
        os.environ["RAG_PERSIST_DIR"] = os.path.join(cls.tmpdir.name, "chroma")
        os.environ["APP_ENV"] = "testing"
        os.environ["LANGSMITH_TRACING"] = "false"
        for name in cls.MODULE_NAMES:
            sys.modules.pop(name, None)
        cls.app_module = importlib.import_module("app")
        cls.client = cls.app_module.app.test_client()

    @classmethod
    def tearDownClass(cls):
        for name in cls.MODULE_NAMES:
            sys.modules.pop(name, None)
            if cls.saved_modules[name] is not None:
                sys.modules[name] = cls.saved_modules[name]
        for name, value in cls.saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        cls.tmpdir.cleanup()

    def as_browser(self, user_id):
        """Pin the class client's workspace id, the way a returning browser has one."""
        with self.client.session_transaction() as flask_session:
            flask_session["user_id"] = user_id

    def collect(self, backend=None, query="message=보증금 문의", url="http://legal:8000"):
        """Stream one turn; `backend` is a /v1/chat body, an exception to raise, or a Mock."""
        ask = backend if isinstance(backend, mock.Mock) else mock.Mock(
            side_effect=backend if isinstance(backend, Exception) else None,
            return_value=turn(backend_answer() if backend is None else backend),
        )
        self.ask = ask
        with mock.patch.dict(os.environ, {"LEGAL_RAG_URL": url}), \
                mock.patch.object(self.app_module.legal_backend, "ask", ask):
            response = self.client.get("/chat?" + query)
            body = response.get_data(as_text=True)
        return [
            json.loads(line[len("data: "):])
            for line in body.splitlines()
            if line.startswith("data: ")
        ]

    def final(self, events):
        return [e for e in events if e["type"] == "final_response"][0]


class ChatStreamingTests(ChatStreamBase):
    def test_stream_emits_known_event_types_and_terminates(self):
        events = self.collect()

        for event in events:
            self.assertIn(event["type"], FRONTEND_EVENT_TYPES)
        self.assertEqual(events[0]["type"], "thought")
        self.assertEqual(events[-2]["type"], "final_response")
        self.assertEqual(events[-1]["type"], "stream_end")

    def test_the_draft_is_forwarded_before_the_answer_and_the_answer_is_unchanged(self):
        drafts = [{"type": "draft_reset"}, {"type": "draft", "text": "분석"}]
        streamed = mock.Mock(return_value=[*drafts, *turn(backend_answer("요약입니다.", "분석입니다."))])

        events = self.collect(streamed)

        kinds = [event["type"] for event in events]
        self.assertLess(kinds.index("draft"), kinds.index("final_response"))
        self.assertIn({"type": "draft_reset"}, events)
        self.assertIn({"type": "draft", "content": "분석"}, events)
        self.assertEqual(self.final(events)["chat_response"], "요약입니다.\n\n분석입니다.")

    def test_each_backend_stage_becomes_a_status_line(self):
        stages = [{"type": "stage", "stage": name} for name in ("analyze", "search", "write", "verify", "rewrite")]
        events = self.collect(mock.Mock(return_value=[*stages, *turn(backend_answer())]))

        self.assertEqual([e["content"] for e in events if e["type"] == "thought"][1:6],
                         ["질문 분석 중", "법령·판례 검색 중", "답변 작성 중", "근거 확인 중", "근거가 맞지 않아 다시 쓰는 중"])

    def test_final_response_carries_the_answer(self):
        final = self.final(self.collect(backend_answer("요약입니다.", "분석입니다.")))

        self.assertEqual(final["chat_response"], "요약입니다.\n\n분석입니다.")
        self.assertEqual(set(final), {"type", "chat_response", "sources", "questions"})

    def test_final_response_lists_the_questions_the_answer_asks_the_user(self):
        body = backend_answer()
        body["answer"]["additional_information_needed"] = ["계약 종료일이 언제인가요?", "보증금은 얼마인가요?"]

        self.assertEqual(self.final(self.collect(body))["questions"], ["계약 종료일이 언제인가요?", "보증금은 얼마인가요?"])
        asked = self.final(self.collect({"kind": "clarification", "questions": ["언제 해고되었나요?"]}))
        self.assertEqual(asked["questions"], ["언제 해고되었나요?"])

    def test_a_statement_is_not_offered_for_answering(self):
        """qwen3:4b sometimes writes a statement in this field; a button for it has no answer (2026-10-06).
        It also writes the fact it needs as a noun phrase, and that one can be answered."""
        body = backend_answer()
        body["answer"]["additional_information_needed"] = ["임대인은 24시간 이내에 반환해야 합니다.", "집을 비워 준 날이 언제인가요?",
                                                           "임대차 계약이 실제로 종료된 날짜"]

        self.assertEqual(self.final(self.collect(body))["questions"], ["집을 비워 준 날이 언제인가요?", "임대차 계약이 실제로 종료된 날짜"])

    def test_an_answer_with_a_deadline_or_an_amount_says_to_check_the_source(self):
        """The model sometimes misuses a figure that is in its sources ("두 달 전" became "2개월 이내"),
        which no text comparison catches (memory.md, 2026-10-06). The reader is told to check."""
        for summary in ("해고일부터 3개월 이내에 신청해야 합니다.", "보증금 1억 2,000만 원을 청구할 수 있습니다.",
                        "2026년 8월 31일에 계약이 끝났습니다.", "연 5%의 지연손해금이 붙습니다.", "두 달 안에 신청하세요."):
            with self.subTest(summary=summary):
                self.assertIn("※ 기간·날짜·금액", self.final(self.collect(backend_answer(summary)))["chat_response"])

    def test_an_answer_without_figures_carries_no_such_notice(self):
        for summary in ("노동위원회에 구제신청을 할 수 있습니다.", "주택임대차보호법 제3조의3 제1항에 따라 신청할 수 있습니다[1]."):
            with self.subTest(summary=summary):
                self.assertNotIn("※ 기간·날짜·금액", self.final(self.collect(backend_answer(summary)))["chat_response"])
        self.assertNotIn("※ 기간·날짜·금액", self.final(self.collect({"kind": "rejected"}))["chat_response"])

    def test_a_refusal_and_a_greeting_ask_nothing(self):
        self.assertEqual(self.final(self.collect({"kind": "rejected", "reason": "unsupported citations"}))["questions"], [])
        self.assertEqual(self.final(self.collect(query="message=안녕하세요"))["questions"], [])

    def test_final_response_carries_the_evidence_the_backend_used(self):
        final = self.final(self.collect(backend_answer("보증금 반환을 청구할 수 있습니다(S1).", items=[STATUTE_ITEM])))

        self.assertEqual(final["sources"], [{"source": "주택임대차보호법 제3조", "kind": "법령", "score": 0.9,
                                             "quote": "임차인은 보증금의 반환을 청구할 수 있다.",
                                             "chunk_id": "law#art:제3조"}])

    def test_the_backend_gets_the_message_and_a_conversation_of_this_browser(self):
        self.collect(query="message=해고당했어요")

        url, message, conversation_id = self.ask.call_args.args[:3]
        with self.client.session_transaction() as flask_session:
            browser_id = flask_session["user_id"]
        self.assertEqual((url, message), ("http://legal:8000", "해고당했어요"))
        self.assertTrue(conversation_id.startswith(browser_id))

    def test_a_greeting_is_answered_without_calling_the_backend(self):
        events = self.collect(query="message=안녕하세요")

        self.ask.assert_not_called()
        self.assertEqual(self.final(events)["sources"], [])
        self.assertEqual(events[-1]["type"], "stream_end")

    def test_questions_about_the_service_are_answered_without_the_backend(self):
        """Sent to the backend they came back as a legal analysis citing unrelated sources (2026-10-13)."""
        expected = {"너는 뭘 할 수 있어?": "법률 상담을 돕는 AI", "넌 AI야?": "법률 상담을 돕는 AI", "변호사랑 뭐가 달라?": "변호사가 아니므로",
                    "여기서 해 주는 답변 믿어도 돼?": "변호사가 검토한 답변이 아닙니다", "이 답변은 변호사가 확인한 거야?": "틀릴 수 있고",
                    "내가 쓴 대화 내용은 저장돼?": "서버가 다시 시작되면", "여기 상담 기록은 누가 볼 수 있어?": "다른 브라우저에서는 열 수 없습니다",
                    "이거 무료야?": "무료로 이용", "사람이랑 직접 얘기하고 싶어": "직접 연결해 드리는 기능이 없습니다"}
        for message, part in expected.items():
            with self.subTest(message=message):
                final = self.final(self.collect(query="message=" + quote(message)))
                self.ask.assert_not_called()
                self.assertIn(part, final["chat_response"])
                self.assertEqual((final["sources"], final["questions"]), ([], []))

    def test_thanks_and_acknowledgements_get_their_own_short_reply(self):
        for message, part in {"고마워요, 도움이 됐어요": "도움이 되었다니", "네 알겠습니다": "이어서 적어 주세요", "ㅋㅋ 알겠어": "이어서 적어 주세요",
                              "안녕하세요, 처음 써봐요": "안녕하세요"}.items():
            with self.subTest(message=message):
                final = self.final(self.collect(query="message=" + quote(message)))
                self.ask.assert_not_called()
                self.assertIn(part, final["chat_response"])

    def test_a_bare_yes_or_no_reaches_the_backend_once_the_consultation_has_an_answer(self):
        """It answers what the consultation asked; before any answer it is only a short message."""
        self.as_browser("yes-no-browser")
        for message in ("네", "아니요"):
            with self.subTest(first=message):
                self.assertIn("안녕하세요", self.final(self.collect(query="message=" + quote(message)))["chat_response"])
                self.ask.assert_not_called()
        for message in ("계약서를 쓰긴 했어요", "네", "아니요", "응."):
            with self.subTest(later=message):
                self.client.post("/chat", json={"message": message})
                self.collect(query="message=" + quote(message))
                self.ask.assert_called_once()
        self.client.post("/chat", json={"message": "ㅇㅇ"})
        self.assertIn("이어서 적어 주세요", self.final(self.collect(query="message=" + quote("ㅇㅇ")))["chat_response"])
        self.ask.assert_not_called()

    def test_legal_questions_that_share_words_with_those_still_reach_the_backend(self):
        for message in ("소송 비용은 누가 내?", "판결문은 믿어도 되는 증거야?", "녹음 파일은 저장해 둬야 해?", "변호사 없이 혼자 소송할 수 있어?",
                        "법률구조공단은 어떻게 이용해?", "고마워. 근데 변호사 없이도 할 수 있어?", "감사합니다 근데 항소는요", "네, 근로계약서는 썼어요",
                        "알겠어요 그럼 소장은요", "무료 법률 상담은 어디서 받아?", "요즘 집주인 때문에 너무 스트레스 받아"):
            with self.subTest(message=message):
                self.collect(query="message=" + quote(message))
                self.ask.assert_called_once()

    def test_a_clarification_shows_the_backends_questions(self):
        final = self.final(self.collect({"kind": "clarification", "questions": ["언제 해고되었나요?"]}))

        self.assertIn("언제 해고되었나요?", final["chat_response"])

    def test_a_rejected_answer_says_it_could_not_be_grounded(self):
        final = self.final(self.collect({"kind": "rejected", "reason": "unsupported citations"}))

        self.assertIn("근거를 확인할 수 없어", final["chat_response"])

    def test_an_unavailable_backend_is_an_error_not_an_answer(self):
        events = self.collect(self.app_module.legal_backend.LegalBackendUnavailable("ConnectError"))

        self.assertEqual([e["type"] for e in events], ["thought", "error", "stream_end"])
        self.assertIn("사용할 수 없습니다", events[1]["content"])

    def test_without_a_configured_backend_the_stream_says_so(self):
        events = self.collect(url="")

        self.ask.assert_not_called()
        self.assertEqual([e["type"] for e in events], ["error", "stream_end"])
        self.assertIn("LEGAL_RAG_URL", events[0]["content"])

    def test_the_reply_is_stored_in_the_chat_history(self):
        self.as_browser("history-browser")
        self.collect(backend_answer("저장되는 답변입니다."))

        history = self.client.get("/get_initial_state").json["chat_history"]
        self.assertEqual(history[-1], {"role": "assistant", "content": "저장되는 답변입니다."})

    def test_the_reply_is_stored_when_the_browser_leaves_during_the_draft(self):
        """Before the draft streamed, the reply was stored before anything after the status lines was
        sent; a browser that reloads mid-draft must still find the answer in its history."""
        self.as_browser("leaving-browser")
        backend = [{"type": "draft_reset"}, {"type": "draft", "text": "분석"}, *turn(backend_answer("떠나도 저장되는 답변입니다."))]
        with mock.patch.dict(os.environ, {"LEGAL_RAG_URL": "http://legal:8000"}), \
                mock.patch.object(self.app_module.legal_backend, "ask", mock.Mock(return_value=iter(backend))):
            response = self.client.get("/chat?message=보증금 문의", buffered=False)
            for chunk in response.response:
                if b'"draft"' in chunk:
                    break
            response.close()

        history = self.client.get("/get_initial_state").json["chat_history"]
        self.assertEqual(history[-1], {"role": "assistant", "content": "떠나도 저장되는 답변입니다."})

    def test_missing_message_still_ends_the_stream(self):
        response = self.client.get("/chat")
        body = response.get_data(as_text=True)
        types = [
            json.loads(line[len("data: "):])["type"]
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual(types, ["error", "stream_end"])


class LegalSourceTests(ChatStreamBase):
    """The reader a source card opens asks this route for the whole document."""

    DOCUMENT = {"before": "제2조 ...\n\n", "cited": "제3조 임차인은 보증금의 반환을 청구할 수 있다.", "after": ""}

    def read(self, document=DOCUMENT, query="chunk_id=law%23art:제3조", url="http://legal:8000"):
        self.lookup = mock.Mock(side_effect=document if isinstance(document, Exception) else None, return_value=document)
        with mock.patch.dict(os.environ, {"LEGAL_RAG_URL": url}), \
                mock.patch.object(self.app_module.legal_backend, "document", self.lookup):
            return self.client.get("/legal_source?" + query)

    def test_the_document_and_the_cited_place_are_returned(self):
        response = self.read()

        self.assertEqual((response.status_code, response.json), (200, self.DOCUMENT))
        self.lookup.assert_called_once_with("http://legal:8000", "law#art:제3조")

    def test_a_source_that_is_not_indexed_is_not_found(self):
        self.assertEqual(self.read(None).status_code, 404)

    def test_an_unavailable_backend_is_a_gateway_error(self):
        response = self.read(self.app_module.legal_backend.LegalBackendUnavailable("ConnectError"))

        self.assertEqual(response.status_code, 502)
        self.assertNotIn("ConnectError", response.json["error"])

    def test_a_request_without_a_chunk_or_a_backend_does_not_call_out(self):
        self.assertEqual(self.read(query="").status_code, 400)
        self.assertEqual(self.read(url="").status_code, 503)
        self.lookup.assert_not_called()


class CaseSearchPageTests(ChatStreamBase):
    """The case-search page searches cases and reads the article a filter names through these routes."""

    def backend(self, **stubs):
        patches = [mock.patch.dict(os.environ, {"LEGAL_RAG_URL": stubs.pop("url", "http://legal:8000")})]
        self.calls = {name: mock.Mock(side_effect=value if isinstance(value, Exception) else None, return_value=value)
                      for name, value in stubs.items()}
        patches += [mock.patch.object(self.app_module.legal_backend, name, call) for name, call in self.calls.items()]
        return patches

    def get(self, path, **stubs):
        with contextlib.ExitStack() as stack:
            for patch in self.backend(**stubs):
                stack.enter_context(patch)
            return self.client.post(path[0], json=path[1]) if isinstance(path, tuple) else self.client.get(path)

    def test_a_law_is_returned_with_its_articles(self):
        law = {"name": "주택임대차보호법", "articles": [{"article": "제3조", "title": "대항력 등", "text": "...", "chunk_id": "c"}]}

        self.assertEqual(self.get("/law?name=주택임대차보호법", law=law).json, law)
        self.assertEqual(self.get("/law?name=없는법", law=None).status_code, 404)

    def test_cases_are_searched_with_the_query_the_page_built(self):
        cards = [{"title": "대법원 2019다1", "subject": "임대차보증금", "date": "2019.05.30", "kind": "판례",
                  "quote": "...", "score": 0.9, "chunk_id": "c",
                  "statutes": [{"law": "주택임대차보호법", "article": "제3조의3", "paragraph": "제1항"}]}]

        response = self.get(("/case_search", {"query": "임차권등기명령 보증금 반환"}), search_cases=cards)

        self.assertEqual((response.status_code, response.json), (200, {"results": cards}))
        self.calls["search_cases"].assert_called_once_with("http://legal:8000", "임차권등기명령 보증금 반환")

    def test_an_empty_request_or_a_missing_backend_does_not_call_out(self):
        self.assertEqual(self.get("/law?name=", law={}).status_code, 400)
        self.assertEqual(self.get(("/case_search", {"query": "  "}), search_cases=[]).status_code, 400)
        self.assertEqual(self.get(("/case_search", {"query": ["x"]}), search_cases=[]).status_code, 400)
        self.assertEqual(self.get("/law?name=민법", law={}, url="").status_code, 503)
        self.calls["law"].assert_not_called()

    def test_an_unavailable_backend_is_a_gateway_error(self):
        down = self.app_module.legal_backend.LegalBackendUnavailable("ConnectError")

        self.assertEqual(self.get(("/case_search", {"query": "보증금"}), search_cases=down).status_code, 502)
        self.assertEqual(self.get("/law?name=민법", law=down).status_code, 502)


if __name__ == "__main__":
    unittest.main()

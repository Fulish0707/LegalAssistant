"""SSE contract tests for /chat.

The subject here is the event contract static/lc-chat.js consumes, not answer
quality: every test stubs legal_backend.ask, the call to legal_rag's /v1/chat.
A real call would need the service, its indexes and a model.
"""

import importlib
import json
import os
import sys
import tempfile
import unittest
from unittest import mock


# Exactly the set static/lc-chat.js branches on (sendMessage, sse.onmessage).
FRONTEND_EVENT_TYPES = {
    "thought",
    "final_response",
    "error",
    "stream_end",
}


def backend_answer(summary="보증금은 돌려받을 수 있습니다.", analysis="", items=()):
    """A /v1/chat body of kind "answer"."""
    return {
        "kind": "answer",
        "answer": {"summary": summary, "analysis": analysis, "possible_actions": [],
                   "additional_information_needed": [], "uncertainty": "", "confidence": "medium"},
        "evidence": {"items": list(items)},
    }


STATUTE_ITEM = {"source_id": "S1", "document_type": "statute", "score": 0.9,
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
            return_value=backend_answer() if backend is None else backend,
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

    def test_final_response_carries_the_answer(self):
        final = self.final(self.collect(backend_answer("요약입니다.", "분석입니다.")))

        self.assertEqual(final["chat_response"], "요약입니다.\n\n분석입니다.")
        self.assertEqual(set(final), {"type", "chat_response", "sources"})

    def test_final_response_carries_the_evidence_the_backend_used(self):
        final = self.final(self.collect(backend_answer("보증금 반환을 청구할 수 있습니다(S1).", items=[STATUTE_ITEM])))

        self.assertEqual(final["sources"], [{"source": "주택임대차보호법 제3조", "kind": "법령", "score": 0.9,
                                             "quote": "임차인은 보증금의 반환을 청구할 수 있다."}])

    def test_the_backend_gets_the_message_and_a_conversation_of_this_browser(self):
        self.collect(query="message=해고당했어요")

        url, message, conversation_id = self.ask.call_args.args
        with self.client.session_transaction() as flask_session:
            browser_id = flask_session["user_id"]
        self.assertEqual((url, message), ("http://legal:8000", "해고당했어요"))
        self.assertTrue(conversation_id.startswith(browser_id))

    def test_a_greeting_is_answered_without_calling_the_backend(self):
        events = self.collect(query="message=안녕하세요")

        self.ask.assert_not_called()
        self.assertEqual(self.final(events)["sources"], [])
        self.assertEqual(events[-1]["type"], "stream_end")

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

    def test_missing_message_still_ends_the_stream(self):
        response = self.client.get("/chat")
        body = response.get_data(as_text=True)
        types = [
            json.loads(line[len("data: "):])["type"]
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual(types, ["error", "stream_end"])


if __name__ == "__main__":
    unittest.main()

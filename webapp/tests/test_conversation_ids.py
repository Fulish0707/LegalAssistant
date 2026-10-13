"""Each browser talks to legal_rag under its own conversation id.

The id is built from the opaque workspace id in the session cookie, so two browsers
never share a conversation and one browser returns to its own. A new consultation and
a cleared chat start a new id: legal_rag keeps recent turns per id and has no call to
forget them. Earlier consultations stay listed and can be reopened.
"""

import os
import tempfile
from unittest.mock import Mock, patch

from tests.test_chat_streaming import STATUTE_ITEM, ChatStreamBase, backend_answer


class ConversationIdTests(ChatStreamBase):
    def conversation_of(self, client):
        ask = Mock(return_value={"kind": "rejected"})
        with patch.dict(os.environ, {"LEGAL_RAG_URL": "http://legal:8000"}), \
                patch.object(self.app_module.legal_backend, "ask", ask):
            client.get("/chat?message=보증금 문의").get_data()
        with client.session_transaction() as flask_session:
            return flask_session["user_id"], ask.call_args.args[2]

    def test_a_conversation_is_keyed_on_the_browsers_own_id(self):
        browser_id, conversation = self.conversation_of(self.app_module.app.test_client())
        self.assertTrue(browser_id and conversation.startswith(browser_id))

    def test_two_browsers_never_share_a_conversation(self):
        _, first = self.conversation_of(self.app_module.app.test_client())
        _, second = self.conversation_of(self.app_module.app.test_client())
        self.assertNotEqual(first, second)

    def test_the_same_browser_returns_to_the_same_conversation(self):
        client = self.app_module.app.test_client()
        self.assertEqual(self.conversation_of(client)[1], self.conversation_of(client)[1])

    def test_clearing_chat_history_starts_a_new_conversation(self):
        client = self.app_module.app.test_client()
        _, before = self.conversation_of(client)
        self.assertEqual(client.post("/clear_chat_history_endpoint").status_code, 200)
        self.assertNotEqual(before, self.conversation_of(client)[1])


class ConsultationListTests(ChatStreamBase):
    """새 상담 keeps the consultations before it; each can be reopened with its own messages."""

    def ask(self, client, message, body=None):
        client.post("/chat", json={"message": message})
        with patch.dict(os.environ, {"LEGAL_RAG_URL": "http://legal:8000"}), \
                patch.object(self.app_module.legal_backend, "ask", Mock(return_value=body or {"kind": "rejected"})) as backend:
            client.get("/chat", query_string={"message": message}).get_data()
        return backend.call_args.args[2]

    def test_a_new_consultation_is_empty_and_the_earlier_one_stays_listed(self):
        client = self.app_module.app.test_client()
        self.ask(client, "전세 보증금을 못 받았어요\n집주인이 연락을 받지 않습니다")

        state = client.post("/conversations/new").json

        self.assertEqual(state["chat_history"], [])
        self.assertEqual([c["title"] for c in state["conversations"]], ["전세 보증금을 못 받았어요"])
        self.assertNotEqual(state["conversation"], state["conversations"][0]["id"])

    def test_each_consultation_keeps_its_own_messages_and_backend_conversation(self):
        client = self.app_module.app.test_client()
        first_id = self.ask(client, "전세 보증금을 못 받았어요")
        client.post("/conversations/new")
        second_id = self.ask(client, "부당해고를 당했어요")

        state = client.get("/get_initial_state").json
        self.assertEqual([m["content"] for m in state["chat_history"] if m["role"] == "user"], ["부당해고를 당했어요"])
        self.assertEqual([c["title"] for c in state["conversations"]], ["부당해고를 당했어요", "전세 보증금을 못 받았어요"])
        self.assertNotEqual(first_id, second_id)

        reopened = client.post("/conversations/select", json={"id": state["conversations"][1]["id"]}).json
        self.assertEqual([m["content"] for m in reopened["chat_history"] if m["role"] == "user"], ["전세 보증금을 못 받았어요"])
        self.assertEqual(self.ask(client, "언제까지 기다려야 하나요"), first_id)

    def test_a_reopened_consultation_shows_the_sources_of_its_last_answer(self):
        client = self.app_module.app.test_client()
        self.ask(client, "보증금을 돌려받을 수 있나요", backend_answer("청구할 수 있습니다(S1).", items=[STATUTE_ITEM]))
        cited = client.get("/get_initial_state").json["sources"]
        self.assertEqual([(card["source"], card["chunk_id"]) for card in cited], [("주택임대차보호법 제3조", STATUTE_ITEM["chunk_id"])])

        self.assertEqual(client.post("/conversations/new").json["sources"], [])
        self.ask(client, "부당해고를 당했어요")
        self.assertEqual(client.get("/get_initial_state").json["sources"], [])

        first = client.get("/get_initial_state").json["conversations"][-1]["id"]
        self.assertEqual(client.post("/conversations/select", json={"id": first}).json["sources"], cited)

    def test_a_failed_turn_keeps_the_sources_of_the_answer_before_it(self):
        client = self.app_module.app.test_client()
        self.ask(client, "보증금을 돌려받을 수 있나요", backend_answer("청구할 수 있습니다(S1).", items=[STATUTE_ITEM]))
        cited = client.get("/get_initial_state").json["sources"]

        client.post("/chat", json={"message": "언제까지 기다려야 하나요"})
        with patch.dict(os.environ, {"LEGAL_RAG_URL": "http://legal:8000"}), \
                patch.object(self.app_module.legal_backend, "ask", Mock(side_effect=self.app_module.legal_backend.LegalBackendUnavailable("down"))):
            client.get("/chat", query_string={"message": "언제까지 기다려야 하나요"}).get_data()

        self.assertEqual(client.get("/get_initial_state").json["sources"], cited)

    def test_an_empty_consultation_is_reused_instead_of_opening_another(self):
        client = self.app_module.app.test_client()
        self.ask(client, "전세 보증금을 못 받았어요")

        first = client.post("/conversations/new").json["conversation"]

        self.assertEqual(client.post("/conversations/new").json["conversation"], first)

    def test_a_browser_cannot_open_a_consultation_it_does_not_hold(self):
        owner, other = self.app_module.app.test_client(), self.app_module.app.test_client()
        self.ask(owner, "전세 보증금을 못 받았어요")
        owner.post("/conversations/new")
        self.ask(owner, "부당해고를 당했어요")

        self.assertEqual(other.post("/conversations/select", json={"id": 1}).status_code, 404)
        self.assertEqual(other.get("/get_initial_state").json["conversations"], [])
        self.assertEqual(owner.post("/conversations/select", json={"id": 7}).status_code, 404)

    CASE = {"chunk_id": "서울중앙지방법원-2023가단5103393_chunk_0001", "title": "서울중앙지방법원 2023가단5103393"}

    def pinned(self, client, message="이 판례가 제 사건에도 적용될 수 있나요?"):
        client.post("/chat", json={"message": message})
        with patch.dict(os.environ, {"LEGAL_RAG_URL": "http://legal:8000"}), \
                patch.object(self.app_module.legal_backend, "ask", Mock(return_value={"kind": "rejected"})) as backend:
            client.get("/chat", query_string={"message": message}).get_data()
        return list(backend.call_args.kwargs["pinned"])

    def test_an_attached_case_goes_with_every_turn_of_its_consultation_only(self):
        client = self.app_module.app.test_client()

        self.assertEqual(client.post("/conversations/case", json=self.CASE).json["case"], self.CASE)
        self.assertEqual(self.pinned(client), [self.CASE["chunk_id"]])
        self.assertEqual(self.pinned(client, "그럼 제 경우에는 어떤가요?"), [self.CASE["chunk_id"]])

        self.assertIsNone(client.post("/conversations/new").json["case"])
        self.assertEqual(self.pinned(client, "부당해고를 당했어요"), [])

        first = client.get("/get_initial_state").json["conversations"][-1]["id"]
        self.assertEqual(client.post("/conversations/select", json={"id": first}).json["case"], self.CASE)

    def test_a_detached_case_no_longer_goes_with_the_turn(self):
        client = self.app_module.app.test_client()
        client.post("/conversations/case", json=self.CASE)

        self.assertIsNone(client.post("/conversations/case", json={"chunk_id": None}).json["case"])
        self.assertEqual(self.pinned(client), [])

    def test_a_case_attached_before_a_server_start_is_gone_after_it(self):
        client = self.app_module.app.test_client()
        client.post("/conversations/case", json=self.CASE)

        with patch.object(self.app_module, "SERVER_START", "restarted"):
            self.assertIsNone(client.get("/get_initial_state").json["case"])
            self.assertEqual(self.pinned(client), [])

    def test_deleting_the_open_consultation_removes_only_it_and_opens_an_empty_one(self):
        client, other = self.app_module.app.test_client(), self.app_module.app.test_client()
        self.ask(other, "다른 브라우저의 상담")
        first_id = self.ask(client, "전세 보증금을 못 받았어요")
        client.post("/conversations/new")
        self.ask(client, "부당해고를 당했어요", backend_answer("구제신청을 할 수 있습니다(S1).", items=[STATUTE_ITEM]))
        client.post("/conversations/case", json=self.CASE)

        state = client.post("/conversations/delete").json

        self.assertEqual((state["chat_history"], state["sources"], state["case"]), ([], [], None))
        self.assertEqual([c["title"] for c in state["conversations"]], ["전세 보증금을 못 받았어요"])
        self.assertEqual(len(other.get("/get_initial_state").json["chat_history"]), 2)
        # The deleted consultation's number is not given out again: the backend still remembers its turns.
        self.assertNotIn(self.ask(client, "새로 묻습니다"), {first_id, first_id[:-1] + "1"})
        self.assertEqual(len(client.get("/get_initial_state").json["conversations"]), 2)

    def test_deleting_an_empty_consultation_changes_nothing_that_is_stored(self):
        client = self.app_module.app.test_client()
        self.ask(client, "전세 보증금을 못 받았어요")
        client.post("/conversations/new")

        state = client.post("/conversations/delete").json

        self.assertEqual([c["title"] for c in state["conversations"]], ["전세 보증금을 못 받았어요"])
        self.assertEqual(state["chat_history"], [])

    def test_a_server_start_forgets_every_consultation(self):
        """`python webapp/app.py` calls this before it serves. The browser's cookie outlives the
        start, so the backend conversation must change too."""
        client = self.app_module.app.test_client()
        before = self.ask(client, "전세 보증금을 못 받았어요", backend_answer("청구할 수 있습니다(S1).", items=[STATUTE_ITEM]))

        self.app_module.clear_all_chat_history()
        with patch.object(self.app_module, "SERVER_START", "restarted"):
            state = client.get("/get_initial_state").json
            after = self.ask(client, "부당해고를 당했어요")

        self.assertEqual((state["chat_history"], state["conversations"], state["sources"]), ([], [], []))
        self.assertNotEqual(before, after)

    def test_messages_stored_before_consultations_existed_form_the_first_one(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "old.db")
            old = sqlite3.connect(path)
            old.execute("CREATE TABLE chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, "
                        "role TEXT NOT NULL, content TEXT NOT NULL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
            old.execute("INSERT INTO chat_history (user_id, role, content) VALUES ('u', 'user', '예전 질문')")
            old.commit()

            self.app_module.ensure_database_schema(old)

            self.assertEqual(old.execute("SELECT content, conversation, sources FROM chat_history").fetchall(), [("예전 질문", 0, None)])
            old.close()

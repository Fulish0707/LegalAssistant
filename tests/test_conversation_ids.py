"""Each browser talks to legal_rag under its own conversation id.

The id is built from the opaque workspace id in the session cookie, so two browsers
never share a conversation and one browser returns to its own. Clearing the chat
starts a new id: legal_rag keeps recent turns per id and has no call to forget them.
"""

import os
from unittest.mock import Mock, patch

from tests.test_chat_streaming import ChatStreamBase


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

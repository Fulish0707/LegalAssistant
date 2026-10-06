"""Conversation memory: a LangGraph checkpointer, keyed by thread_id.

An earlier branch deleted ConversationBufferMemory and nothing replaced it, so the
agent could not remember anything said on a previous turn. A checkpointer restores
that, and brings a new isolation surface with it: thread_id decides whose messages
come back, exactly as the user_id filter does on the retrieval side
(tests/test_rag_isolation.py). The two threads test below is the memory-side
counterpart to that guard.

Almost everything here runs against a fake chat model. The property under test is
that state is persisted per thread and isolated between threads -- not that the LLM
says anything in particular -- and a real qwen3:4b turn costs tens of seconds. One
live test is kept, and it skips when Ollama is unreachable.
"""

import importlib
import os
import sys
import tempfile
import unittest
from typing import Any
from unittest.mock import Mock, patch

import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeChatModel(BaseChatModel):
    """A model that answers instantly and never leaves the process.

    create_agent compiles the same graph around it that it compiles around
    ChatOllama: bind_tools is what it needs, and a final message with no tool calls
    ends the loop. Verified: the graph terminates and the checkpointer records both
    the human and the AI message.
    """

    reply: str = "ack"

    @property
    def _llm_type(self):
        return "fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])


def build_fake_agent(user_id, checkpointer, reply="ack"):
    import langchain_config

    with patch.object(langchain_config, "get_chat_model", return_value=FakeChatModel(reply=reply)):
        return langchain_config.build_agent(user_id, checkpointer=checkpointer)


def thread_texts(agent, thread_id):
    state = agent.get_state({"configurable": {"thread_id": thread_id}})
    return [message.content for message in state.values.get("messages", [])]


class AgentMemoryTests(unittest.TestCase):
    def test_checkpointer_persists_turns_per_thread(self):
        from langgraph.checkpoint.sqlite import SqliteSaver

        with tempfile.TemporaryDirectory() as tmp:
            with SqliteSaver.from_conn_string(f"{tmp}/checkpoints.db") as saver:
                agent = build_fake_agent("user-1", saver)
                config = {"configurable": {"thread_id": "user-1"}}

                agent.invoke({"messages": [("user", "My name is Ada.")]}, config)
                state = agent.get_state(config)

                self.assertGreater(len(state.values["messages"]), 1)

    def test_a_second_turn_sees_what_the_first_turn_said(self):
        """The point of the checkpointer. Without it the second invoke starts from
        an empty message list and the first turn is gone."""
        from langgraph.checkpoint.sqlite import SqliteSaver

        with tempfile.TemporaryDirectory() as tmp:
            with SqliteSaver.from_conn_string(f"{tmp}/checkpoints.db") as saver:
                agent = build_fake_agent("user-1", saver)
                config = {"configurable": {"thread_id": "user-1"}}

                agent.invoke({"messages": [("user", "My name is Ada.")]}, config)
                second = agent.invoke({"messages": [("user", "What is my name?")]}, config)

                contents = [message.content for message in second["messages"]]
                self.assertIn("My name is Ada.", contents)
                self.assertIn("What is my name?", contents)

    def test_threads_are_isolated_between_users(self):
        """Write to both threads, so a read that ignored thread_id would fail.

        The obvious version of this test -- write one thread, assert the other is
        empty -- passes even if thread_id were ignored on read, because an untouched
        thread is empty either way.
        """
        from langgraph.checkpoint.sqlite import SqliteSaver

        with tempfile.TemporaryDirectory() as tmp:
            with SqliteSaver.from_conn_string(f"{tmp}/checkpoints.db") as saver:
                agent = build_fake_agent("user-1", saver)

                agent.invoke(
                    {"messages": [("user", "My name is Ada.")]},
                    {"configurable": {"thread_id": "user-1"}},
                )
                agent.invoke(
                    {"messages": [("user", "My name is Grace.")]},
                    {"configurable": {"thread_id": "user-2"}},
                )

                first = thread_texts(agent, "user-1")
                second = thread_texts(agent, "user-2")

                self.assertIn("My name is Ada.", first)
                self.assertNotIn("My name is Grace.", first)
                self.assertIn("My name is Grace.", second)
                self.assertNotIn("My name is Ada.", second)

    def test_an_untouched_thread_starts_empty(self):
        from langgraph.checkpoint.sqlite import SqliteSaver

        with tempfile.TemporaryDirectory() as tmp:
            with SqliteSaver.from_conn_string(f"{tmp}/checkpoints.db") as saver:
                agent = build_fake_agent("user-1", saver)

                agent.invoke(
                    {"messages": [("user", "My name is Ada.")]},
                    {"configurable": {"thread_id": "user-1"}},
                )
                other = agent.get_state({"configurable": {"thread_id": "user-2"}})

                self.assertEqual(other.values.get("messages", []), [])


class AppCheckpointerWiringTests(unittest.TestCase):
    """A browser's conversation is keyed on its own workspace id, and nothing else.

    Every visitor owns one opaque session-cookie user_id. The conversation id sent
    to legal_rag is built from it, so two browsers provably never share a
    conversation and one browser always returns to its own.
    """

    MODULE_NAMES = ("app", "langchain_config", "llm_model")
    PINNED_ENV = (
        "APP_STATE_DB",
        "AGENT_CHECKPOINT_DB",
        "RAG_PERSIST_DIR",
        "APP_ENV",
        "LLM_PROVIDER",
        "EMBEDDING_PROVIDER",
        "LANGSMITH_TRACING",
    )

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.saved_env = {name: os.environ.get(name) for name in cls.PINNED_ENV}
        cls.saved_modules = {name: sys.modules.get(name) for name in cls.MODULE_NAMES}
        os.environ["APP_STATE_DB"] = os.path.join(cls.tmpdir.name, "app_state.db")
        os.environ["AGENT_CHECKPOINT_DB"] = os.path.join(cls.tmpdir.name, "checkpoints.db")
        os.environ["RAG_PERSIST_DIR"] = os.path.join(cls.tmpdir.name, "chroma")
        os.environ["APP_ENV"] = "testing"
        os.environ["LLM_PROVIDER"] = "none"
        os.environ["EMBEDDING_PROVIDER"] = "none"
        os.environ["LANGSMITH_TRACING"] = "false"
        for name in cls.MODULE_NAMES:
            sys.modules.pop(name, None)
        cls.app_module = importlib.import_module("app")

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

    def browser(self):
        """A test client with an established workspace, and that workspace's id."""
        client = self.app_module.app.test_client()
        client.get("/health")
        with client.session_transaction() as flask_session:
            return client, flask_session["user_id"]

    def drive_chat(self, client=None):
        """Stream one turn as a normal browser; return its id and the conversation id sent to legal_rag."""
        if client is None:
            client, browser_id = self.browser()
        else:
            with client.session_transaction() as flask_session:
                browser_id = flask_session["user_id"]
        ask = Mock(return_value={"kind": "rejected"})
        with patch.dict(os.environ, {"LEGAL_RAG_URL": "http://legal:8000"}), \
                patch.object(self.app_module.legal_backend, "ask", ask):
            client.get("/chat?message=보증금 문의").get_data()

        self.assertTrue(ask.called, "the legal backend was never called")
        return browser_id, ask.call_args.args[2]

    def test_every_browser_converses_under_its_own_id(self):
        browser_id, conversation_id = self.drive_chat()

        self.assertTrue(browser_id)
        self.assertTrue(conversation_id.startswith(browser_id))

    def test_two_browsers_never_share_a_conversation(self):
        first_id, first_conversation = self.drive_chat()
        second_id, second_conversation = self.drive_chat()

        self.assertNotEqual(first_id, second_id)
        self.assertNotEqual(first_conversation, second_conversation)

    def test_the_same_browser_returns_to_the_same_conversation(self):
        client, _ = self.browser()

        _, first_conversation = self.drive_chat(client)
        _, second_conversation = self.drive_chat(client)

        self.assertEqual(first_conversation, second_conversation)

    def test_clearing_chat_history_starts_a_new_backend_conversation(self):
        """legal_rag keeps recent turns per conversation id; a cleared chat must not reuse them."""
        client, _ = self.browser()

        _, before = self.drive_chat(client)
        self.assertEqual(client.post("/clear_chat_history_endpoint").status_code, 200)
        _, after = self.drive_chat(client)

        self.assertNotEqual(before, after)

    def test_clearing_chat_history_also_clears_the_agents_thread(self):
        """A user who clears their history and then finds the agent still
        remembering would reasonably call that a bug."""
        client, browser_id = self.browser()

        deleted = []
        with patch.object(self.app_module.CHECKPOINTER, "delete_thread", deleted.append):
            response = client.post("/clear_chat_history_endpoint")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(deleted, [browser_id])

    def test_clearing_still_succeeds_when_the_checkpointer_fails(self):
        """Clearing the display record is the endpoint's primary job."""
        client, _ = self.browser()
        client.post("/chat", json={"message": "remember this"})

        def explode(thread_id):
            raise RuntimeError("checkpoint store is unavailable")

        with patch.object(self.app_module.CHECKPOINTER, "delete_thread", explode):
            response = client.post("/clear_chat_history_endpoint")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(client.get("/get_initial_state").json["chat_history"], [])

    def test_the_apps_own_checkpointer_stores_recalls_and_forgets_a_thread(self):
        """End to end on the real CHECKPOINTER, with nothing patched.

        Every other test here stubs build_agent or delete_thread, so none of them
        would notice if build_checkpointer() itself regressed -- a closed
        connection, a bad path, a missing check_same_thread. This one drives the
        app's actual instance, and it checks that clearing the display history
        leaves the thread genuinely empty rather than merely calling delete_thread.
        """
        client, browser_id = self.browser()
        agent = build_fake_agent(browser_id, self.app_module.CHECKPOINTER)
        config = {"configurable": {"thread_id": browser_id}}

        agent.invoke({"messages": [("user", "My name is Ada.")]}, config)
        second = agent.invoke({"messages": [("user", "What is my name?")]}, config)
        self.assertIn("My name is Ada.", [m.content for m in second["messages"]])

        response = client.post("/clear_chat_history_endpoint")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(thread_texts(agent, browser_id), [])

    def test_the_checkpoint_database_is_configurable_and_separate_from_app_state(self):
        self.assertEqual(
            self.app_module.CHECKPOINT_DB_FILE,
            os.environ["AGENT_CHECKPOINT_DB"],
        )
        self.assertNotEqual(self.app_module.CHECKPOINT_DB_FILE, self.app_module.DB_FILE)


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
PROBE_TIMEOUT_SECONDS = 1.0


def _installed_models():
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=PROBE_TIMEOUT_SECONDS)
        response.raise_for_status()
        return [model.get("name", "") for model in response.json().get("models", [])]
    except Exception:
        return None


class AgentMemoryLiveTests(unittest.TestCase):
    """One real turn, to prove the checkpointer works against the real model too.

    One turn, not two: a second qwen3:4b turn through the tool loop would dominate
    the suite's runtime for a property the fake-model tests already cover.
    """

    def setUp(self):
        self._saved = {name: os.environ.get(name) for name in ("LLM_PROVIDER",)}
        models = _installed_models()
        if models is None:
            self.skipTest(f"Ollama is not reachable at {OLLAMA_BASE_URL}")
        if not any(name == CHAT_MODEL or name.split(":")[0] == CHAT_MODEL.split(":")[0] for name in models):
            self.skipTest(f"Ollama model {CHAT_MODEL} is not pulled (have: {models})")
        os.environ["LLM_PROVIDER"] = "ollama"
        # Keep the turn off the network beyond Ollama itself.

    def tearDown(self):
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_a_real_turn_is_checkpointed(self):
        from langgraph.checkpoint.sqlite import SqliteSaver
        import langchain_config

        with tempfile.TemporaryDirectory() as tmp:
            with SqliteSaver.from_conn_string(f"{tmp}/checkpoints.db") as saver:
                agent = langchain_config.build_agent("user-live", checkpointer=saver)
                if agent is None:
                    self.skipTest("No model configured")

                config = {"configurable": {"thread_id": "user-live"}}
                agent.invoke(
                    {"messages": [("user", "My name is Ada. Reply in one short sentence.")]},
                    config,
                    context=langchain_config.AgentContext(user_id="user-live"),
                )
                state = agent.get_state(config)

                self.assertGreater(len(state.values["messages"]), 1)
                self.assertIn("My name is Ada.", state.values["messages"][0].content)


if __name__ == "__main__":
    unittest.main()

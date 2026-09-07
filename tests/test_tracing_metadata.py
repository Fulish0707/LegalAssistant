"""LangSmith tracing metadata: what a run is tagged with, and that it arrives.

run_metadata() only matters if it reaches LangChain. The wiring test below
captures the runnable config app.py hands to agent.stream() and asserts the
metadata is in it -- offline, with no LangSmith account and no live model.
"""

import importlib
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

from langchain_core.messages import AIMessage, AIMessageChunk


TRACING_ENV = ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT")


class TracingMetadataTests(unittest.TestCase):
    PINNED_ENV = TRACING_ENV + ("LLM_PROVIDER", "EMBEDDING_PROVIDER", "OLLAMA_MODEL")

    def setUp(self):
        self.saved_env = {name: os.environ.get(name) for name in self.PINNED_ENV}

    def tearDown(self):
        for name, value in self.saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_run_metadata_identifies_the_model_provider(self):
        """Provider tagging is what makes Gemini vs Qwen3 comparable in LangSmith."""
        import langchain_config

        os.environ["LLM_PROVIDER"] = "ollama"
        metadata = langchain_config.run_metadata("user-1")

        self.assertEqual(metadata["llm_provider"], "ollama")
        self.assertEqual(metadata["thread_id"], "user-1")
        self.assertIn("embedding_provider", metadata)

    def test_run_metadata_names_the_chat_model(self):
        """Provider alone does not identify a run: qwen3:4b and qwen3:8b differ."""
        import langchain_config

        os.environ["LLM_PROVIDER"] = "ollama"
        os.environ["OLLAMA_MODEL"] = "qwen3:4b"

        self.assertEqual(langchain_config.run_metadata("user-1")["llm_model"], "qwen3:4b")

    def test_run_metadata_tolerates_a_guest_turn(self):
        """Guests have no thread_id. A guest turn must still be traceable."""
        import langchain_config

        os.environ["LLM_PROVIDER"] = "ollama"
        metadata = langchain_config.run_metadata(None)

        self.assertEqual(metadata["thread_id"], "guest")
        self.assertEqual(metadata["llm_provider"], "ollama")

    def test_tracing_is_disabled_by_default(self):
        os.environ.pop("LANGSMITH_TRACING", None)
        import langchain_config

        self.assertFalse(langchain_config.tracing_enabled())

    def test_blank_tracing_flag_does_not_enable_tracing(self):
        """.env.example ships LANGSMITH_TRACING= -- copying it must stay off."""
        import langchain_config

        os.environ["LANGSMITH_TRACING"] = ""
        self.assertFalse(langchain_config.tracing_enabled())
        os.environ["LANGSMITH_TRACING"] = "false"
        self.assertFalse(langchain_config.tracing_enabled())

    def test_tracing_turns_on_when_explicitly_enabled(self):
        import langchain_config

        os.environ["LANGSMITH_TRACING"] = "true"
        self.assertTrue(langchain_config.tracing_enabled())


class _StubAgent:
    """Stands in for a compiled LangGraph agent, recording the config it is given."""

    def __init__(self, events):
        self._events = events
        self.stream_calls = []

    def stream(self, payload, config=None, **kwargs):
        self.stream_calls.append((payload, config, kwargs))
        yield from self._events

    def get_state(self, config):
        raise ValueError("No checkpointer set")


class RunMetadataReachesTheRunTests(unittest.TestCase):
    """The point of the whole task: the config actually carries the metadata."""

    MODULE_NAMES = ("app", "langchain_config", "llm_model", "llm_prompt")

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.saved_env = {
            name: os.environ.get(name)
            for name in ("APP_STATE_DB", "AGENT_CHECKPOINT_DB", "APP_ENV", "LLM_PROVIDER")
        }
        cls.saved_modules = {name: sys.modules.get(name) for name in cls.MODULE_NAMES}
        os.environ["APP_STATE_DB"] = os.path.join(cls.tmpdir.name, "app_state.db")
        os.environ["AGENT_CHECKPOINT_DB"] = os.path.join(cls.tmpdir.name, "checkpoints.db")
        os.environ["APP_ENV"] = "testing"
        os.environ["LLM_PROVIDER"] = "ollama"
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

    def _run_one_turn(self):
        payload = json.dumps({"chat_response": "hi", "statement_suggestion": "",
                              "job_statement_suggestion": "", "report_suggestion": ""})
        agent = _StubAgent(
            [("messages", (AIMessageChunk(content="hi", id="run-1"), {}))]
            + [("values", {"messages": [AIMessage(content=payload)]})]
        )
        with mock.patch.object(self.app_module, "build_agent", return_value=agent):
            self.client.get("/chat?message=hello")
        self.assertTrue(agent.stream_calls, "agent.stream was never called")
        return agent.stream_calls[0][1] or {}

    def test_stream_config_carries_run_metadata(self):
        config = self._run_one_turn()

        metadata = config.get("metadata") or {}
        self.assertEqual(metadata.get("llm_provider"), "ollama")
        self.assertIn("embedding_provider", metadata)
        self.assertIn("llm_model", metadata)

    def test_stream_config_carries_a_provider_tag(self):
        """Tags are what a LangSmith comparison view filters on."""
        config = self._run_one_turn()

        self.assertIn("provider:ollama", config.get("tags") or [])

"""LangSmith tracing metadata: what a run is tagged with.

The web chat no longer runs a LangChain agent (legal_rag's /v1/chat answers), so
no chat turn carries this config any more; the helpers stay with the agent code.
"""

import os
import unittest


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

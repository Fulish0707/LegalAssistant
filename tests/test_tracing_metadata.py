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

    def test_run_metadata_identifies_the_model_provider(self):
        """Provider tagging identifies the connector used for each traced run."""
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

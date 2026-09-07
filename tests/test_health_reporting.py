import importlib
import os
import sys
import tempfile
import unittest


class HealthReportingTests(unittest.TestCase):
    MODULE_NAMES = ("app", "langchain_config", "llm_model")
    PINNED_ENV = (
        "APP_STATE_DB",
        "APP_ENV",
        "LLM_PROVIDER",
        "EMBEDDING_PROVIDER",
        "OLLAMA_BASE_URL",
    )

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.original_env = {name: os.environ.get(name) for name in cls.PINNED_ENV}
        cls.original_modules = {name: sys.modules.get(name) for name in cls.MODULE_NAMES}
        os.environ["APP_STATE_DB"] = os.path.join(cls.tmpdir.name, "app_state.db")
        os.environ["APP_ENV"] = "testing"
        # Pinned to ollama on purpose. Pinning google would make "no probe by
        # default" trivially true and the assertion would prove nothing.
        os.environ["LLM_PROVIDER"] = "ollama"
        os.environ["EMBEDDING_PROVIDER"] = "ollama"
        # A port nothing listens on, so a probe fails fast instead of reporting
        # the developer's own running Ollama.
        os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:1"

        for name in cls.MODULE_NAMES:
            sys.modules.pop(name, None)
        cls.app_module = importlib.import_module("app")
        cls.client = cls.app_module.app.test_client()

    @classmethod
    def tearDownClass(cls):
        for name in cls.MODULE_NAMES:
            sys.modules.pop(name, None)
            if cls.original_modules[name] is not None:
                sys.modules[name] = cls.original_modules[name]
        for name, value in cls.original_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        cls.tmpdir.cleanup()

    def test_health_reports_the_active_providers_and_collection(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["llm_provider"], "ollama")
        self.assertEqual(response.json["embedding_provider"], "ollama")
        self.assertEqual(response.json["collection"], "docs_ollama_nomic")
        self.assertNotEqual(response.json["collection"], "langchain")

    def test_health_does_not_probe_the_llm_by_default(self):
        """get_chat_model() returns a ChatOllama unconditionally, so
        llm_configured says nothing about whether Ollama is up. Reachability is a
        separate field, and the probe is opt-in so /health stays cheap."""
        response = self.client.get("/health")

        self.assertIn("llm_reachable", response.json)
        self.assertIsNone(response.json["llm_reachable"])

    def test_health_reports_an_unreachable_llm_when_the_probe_is_requested(self):
        response = self.client.get("/health?probe=llm")

        self.assertEqual(response.status_code, 200)
        self.assertIs(response.json["llm_reachable"], False)

    def test_health_survives_an_unknown_embedding_provider(self):
        """An endpoint whose job is to make misconfiguration visible must not
        crash on misconfiguration. get_collection_name() raises on an unknown
        provider."""
        original = os.environ.get("EMBEDDING_PROVIDER")
        os.environ["EMBEDDING_PROVIDER"] = "not-a-provider"
        try:
            response = self.client.get("/health")
        finally:
            if original is None:
                os.environ.pop("EMBEDDING_PROVIDER", None)
            else:
                os.environ["EMBEDDING_PROVIDER"] = original

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "ok")
        self.assertIsNone(response.json["collection"])


if __name__ == "__main__":
    unittest.main()

import importlib
import os
import sys
import unittest


class ModelConnectorTests(unittest.TestCase):
    def setUp(self):
        sys.modules.pop("llm_model", None)
        self._saved = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved)
        sys.modules.pop("llm_model", None)

    def test_import_does_not_build_clients(self):
        """Importing must be inert: no I/O, no client construction."""
        os.environ.pop("GOOGLE_API_KEY", None)
        module = importlib.import_module("llm_model")

        self.assertFalse(hasattr(module, "llm"))
        self.assertFalse(hasattr(module, "embedding_model"))

    # Do NOT assert os.environ["GOOGLE_API_KEY"] is None here. pytest runs the whole
    # suite in one process, and any earlier test that imports `app` will have called
    # load_dotenv() and repopulated it. That assertion passes or fails on test
    # ordering, not on anything llm_model does. The grep in Step 6 checks it properly.

    def test_google_chat_model_unavailable_without_key(self):
        os.environ.pop("GOOGLE_API_KEY", None)
        os.environ["LLM_PROVIDER"] = "google"
        module = importlib.import_module("llm_model")

        self.assertIsNone(module.get_chat_model())

    def test_collection_name_is_derived_from_embedding_provider(self):
        """Different embedding providers must never share a Chroma collection."""
        module = importlib.import_module("llm_model")

        os.environ["EMBEDDING_PROVIDER"] = "google"
        google_name = module.get_collection_name()
        os.environ["EMBEDDING_PROVIDER"] = "ollama"
        ollama_name = module.get_collection_name()

        self.assertNotEqual(google_name, ollama_name)

    def test_unknown_provider_raises(self):
        os.environ["LLM_PROVIDER"] = "not-a-provider"
        module = importlib.import_module("llm_model")

        with self.assertRaises(ValueError):
            module.get_chat_model()


if __name__ == "__main__":
    unittest.main()

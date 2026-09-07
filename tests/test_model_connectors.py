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
        module = importlib.import_module("llm_model")

        self.assertFalse(hasattr(module, "llm"))
        self.assertFalse(hasattr(module, "embedding_model"))

    # Assert on attributes, not on os.environ. pytest runs the whole suite in one
    # process, and any earlier test that imports `app` will have called
    # load_dotenv(), so an environment assertion here passes or fails on test
    # ordering rather than on anything llm_model does.

    def test_disabled_provider_yields_no_model(self):
        """Ollama needs no credential, so "none" is the only way to express "AI is
        turned off". The app's graceful-degradation paths depend on it."""
        os.environ["LLM_PROVIDER"] = "none"
        os.environ["EMBEDDING_PROVIDER"] = "none"
        module = importlib.import_module("llm_model")

        self.assertIsNone(module.get_chat_model())
        self.assertIsNone(module.get_embedding_model())

    def test_collection_name_is_derived_from_embedding_provider(self):
        """Different embedding providers must never share a Chroma collection."""
        module = importlib.import_module("llm_model")

        os.environ["EMBEDDING_PROVIDER"] = "ollama"
        os.environ["OLLAMA_EMBED_MODEL"] = "nomic-embed-text"
        nomic = module.get_collection_name()
        os.environ["OLLAMA_EMBED_MODEL"] = "mxbai-embed-large"
        other = module.get_collection_name()

        self.assertNotEqual(nomic, other)
        self.assertNotEqual(nomic, "langchain")

    def test_ollama_embedding_model_tag_spellings_share_one_collection(self):
        """`ollama list` prints nomic-embed-text:latest; the bare name is the same
        model. Both spellings must resolve to one collection."""
        module = importlib.import_module("llm_model")

        os.environ["EMBEDDING_PROVIDER"] = "ollama"
        os.environ["OLLAMA_EMBED_MODEL"] = "nomic-embed-text"
        bare = module.get_collection_name()
        os.environ["OLLAMA_EMBED_MODEL"] = "nomic-embed-text:latest"
        tagged = module.get_collection_name()

        self.assertEqual(bare, tagged)

    def test_descriptor_reports_the_spelling_handed_to_the_sdk(self):
        """Normalisation belongs to the collection lookup only. The descriptor must
        keep reporting what get_embedding_model() actually passes to the SDK."""
        module = importlib.import_module("llm_model")

        os.environ["EMBEDDING_PROVIDER"] = "ollama"
        os.environ["OLLAMA_EMBED_MODEL"] = "nomic-embed-text:latest"

        self.assertEqual(
            module.get_embedding_descriptor(), ("ollama", "nomic-embed-text:latest")
        )

    def test_unknown_provider_raises(self):
        os.environ["LLM_PROVIDER"] = "not-a-provider"
        module = importlib.import_module("llm_model")

        with self.assertRaises(ValueError):
            module.get_chat_model()


if __name__ == "__main__":
    unittest.main()

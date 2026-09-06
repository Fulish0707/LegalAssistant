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

    def test_google_embedding_model_spellings_share_one_collection(self):
        """models/text-embedding-004 and text-embedding-004 are the same embedding
        space. Two collection names for one space means a silent re-embed and
        previously ingested documents becoming invisible."""
        module = importlib.import_module("llm_model")

        os.environ["EMBEDDING_PROVIDER"] = "google"
        os.environ["GOOGLE_EMBED_MODEL"] = "models/text-embedding-004"
        prefixed = module.get_collection_name()
        os.environ["GOOGLE_EMBED_MODEL"] = "text-embedding-004"
        bare = module.get_collection_name()

        self.assertEqual(prefixed, bare)

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

        os.environ["EMBEDDING_PROVIDER"] = "google"
        os.environ["GOOGLE_EMBED_MODEL"] = "text-embedding-004"

        self.assertEqual(module.get_embedding_descriptor(), ("google", "text-embedding-004"))

    def test_unknown_provider_raises(self):
        os.environ["LLM_PROVIDER"] = "not-a-provider"
        module = importlib.import_module("llm_model")

        with self.assertRaises(ValueError):
            module.get_chat_model()


if __name__ == "__main__":
    unittest.main()

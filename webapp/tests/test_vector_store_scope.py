import os
import subprocess
import sys
import tempfile
import unittest

from llm_model import get_collection_name


class VectorStoreScopeTests(unittest.TestCase):
    def setUp(self):
        self._saved = {
            name: os.environ.get(name)
            for name in ("EMBEDDING_PROVIDER", "OLLAMA_EMBED_MODEL")
        }

    def tearDown(self):
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_disabled_and_ollama_use_distinct_collection_names(self):
        """The active connector must use a nonempty, explicitly scoped collection."""
        os.environ["EMBEDDING_PROVIDER"] = "none"
        disabled = get_collection_name()
        os.environ["EMBEDDING_PROVIDER"] = "ollama"
        ollama = get_collection_name()

        self.assertNotEqual(disabled, ollama)
        self.assertNotEqual(ollama, "")
        self.assertNotEqual(ollama, "langchain")

    def test_vector_store_is_opened_on_the_provider_scoped_collection(self):
        """Chroma must receive the configured collection instead of its default."""
        import langchain_config

        os.environ["EMBEDDING_PROVIDER"] = "ollama"
        recorded = {}

        original_chroma = langchain_config.Chroma
        original_get_embedding_model = langchain_config.get_embedding_model

        def fake_chroma(**kwargs):
            recorded.update(kwargs)
            return object()

        langchain_config.Chroma = fake_chroma
        langchain_config.get_embedding_model = lambda: object()
        try:
            langchain_config.get_vector_store()
        finally:
            langchain_config.Chroma = original_chroma
            langchain_config.get_embedding_model = original_get_embedding_model

        self.assertEqual(recorded["collection_name"], get_collection_name())
        self.assertNotEqual(recorded["collection_name"], "langchain")

    def test_importing_langchain_config_touches_no_disk_and_builds_no_client(self):
        """Import must be inert. A module-level Chroma handle or a mkdir at import
        makes the collection choice a function of import order, not of env."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with tempfile.TemporaryDirectory() as temp_dir:
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join(
                filter(None, [project_root, environment.get("PYTHONPATH")])
            )
            environment["EMBEDDING_PROVIDER"] = "ollama"

            probe = (
                "import os, langchain_config as m;"
                "assert not hasattr(m, 'vector_store'), 'module-level vector_store';"
                "assert not hasattr(m, 'embedding_model'), 'module-level embedding_model';"
                "assert not os.path.exists('./chroma_db'), 'import created chroma_db';"
                "assert not os.path.exists('./documents'), 'import created documents';"
                "print('INERT')"
            )
            completed = subprocess.run(
                [sys.executable, "-c", probe],
                cwd=temp_dir,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("INERT", completed.stdout)


if __name__ == "__main__":
    unittest.main()

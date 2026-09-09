"""Live check of the Ollama embedding connector.

Skips cleanly when Ollama is not running or the model is not pulled, so the suite
stays green on machines without it. Checks the default embedding dimensions and explicit collection selection.
"""

import os
import unittest

import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
PROBE_TIMEOUT_SECONDS = 1.0


def _installed_models():
    """Model names Ollama reports, or None when it is unreachable."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=PROBE_TIMEOUT_SECONDS)
        response.raise_for_status()
        return [model.get("name", "") for model in response.json().get("models", [])]
    except Exception:
        return None


class OllamaConnectorLiveTests(unittest.TestCase):
    def setUp(self):
        self._saved = {
            name: os.environ.get(name)
            for name in ("EMBEDDING_PROVIDER", "OLLAMA_EMBED_MODEL", "OLLAMA_BASE_URL")
        }
        models = _installed_models()
        if models is None:
            self.skipTest(f"Ollama is not reachable at {OLLAMA_BASE_URL}")
        # Guard on presence, not just liveness: embedding against a missing model
        # triggers a multi-minute pull rather than a fast failure.
        if not any(name == EMBED_MODEL or name.split(":")[0] == EMBED_MODEL.split(":")[0] for name in models):
            self.skipTest(f"Ollama model {EMBED_MODEL} is not pulled (have: {models})")
        os.environ["EMBEDDING_PROVIDER"] = "ollama"

    def tearDown(self):
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_ollama_embeddings_are_768_dimensional(self):
        from llm_model import get_collection_name, get_embedding_model

        embedding_model = get_embedding_model()
        self.assertIsNotNone(embedding_model)

        vector = embedding_model.embed_query("A short probe sentence.")

        self.assertEqual(len(vector), 768)
        # Matching dimensions do not imply matching embedding spaces.
        self.assertNotEqual(get_collection_name(), "langchain")
        self.assertEqual(get_collection_name(), "docs_ollama_nomic")


if __name__ == "__main__":
    unittest.main()

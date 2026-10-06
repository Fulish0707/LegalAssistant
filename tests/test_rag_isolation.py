import importlib
import os
import sys
import unittest


class FakeVectorStore:
    def __init__(self):
        self.added_documents = []

    def add_documents(self, documents):
        self.added_documents.extend(documents)


class RagIsolationTests(unittest.TestCase):
    PINNED_ENV = ("EMBEDDING_PROVIDER",)

    @classmethod
    def setUpClass(cls):
        # Pin EMBEDDING_PROVIDER so an ambient value cannot make an unpatched
        # call reach a live service. Setting it here (rather than trusting the
        # default) also survives app.py's load_dotenv(), which repopulates the
        # environment from .env partway through the suite.
        cls.original_env = {name: os.environ.get(name) for name in cls.PINNED_ENV}
        os.environ["EMBEDDING_PROVIDER"] = "ollama"
        sys.modules.pop("langchain_config", None)
        cls.module = importlib.import_module("langchain_config")

    @classmethod
    def tearDownClass(cls):
        for name, value in cls.original_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        sys.modules.pop("langchain_config", None)

    def setUp(self):
        # get_vector_store() is the seam now: every function looks it up on the
        # module at call time, so replacing the attribute reaches all of them.
        self.original_get_vector_store = self.module.get_vector_store
        self.store = FakeVectorStore()
        self.module.get_vector_store = lambda: self.store

    def tearDown(self):
        self.module.get_vector_store = self.original_get_vector_store


    def test_ingestion_assigns_a_shared_document_owner_and_timestamp_to_every_chunk(self):
        result = self.module.ingest_text_to_vector_store(
            "First paragraph. " * 200,
            "report.txt",
            metadata={"uploaded_via": "test"},
            user_id="account-a",
        )

        self.assertTrue(result["ok"])
        self.assertIn("document_id", result)
        self.assertTrue(result["document_id"])
        self.assertIn("ingested_at", result)
        self.assertTrue(result["ingested_at"].endswith("+00:00"))
        self.assertGreater(result["chunk_count"], 1)
        self.assertTrue(all(chunk.metadata["user_id"] == "account-a" for chunk in self.store.added_documents))
        self.assertTrue(all(chunk.metadata["document_id"] == result["document_id"] for chunk in self.store.added_documents))
        self.assertTrue(all(chunk.metadata["ingested_at"] == result["ingested_at"] for chunk in self.store.added_documents))


if __name__ == "__main__":
    unittest.main()

import importlib
import os
import sys
import unittest


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents

    def invoke(self, query):
        return self.documents


class FakeVectorStore:
    def __init__(self):
        self.retriever_filters = []
        self.added_documents = []

    def as_retriever(self, search_kwargs):
        self.retriever_filters.append(search_kwargs["filter"])
        return FakeRetriever([])

    def add_documents(self, documents):
        self.added_documents.extend(documents)


class RagIsolationTests(unittest.TestCase):
    PINNED_ENV = ("GOOGLE_API_KEY", "EMBEDDING_PROVIDER")

    @classmethod
    def setUpClass(cls):
        # Popping GOOGLE_API_KEY is not enough: any earlier test that imports app
        # has run load_dotenv(), which repopulates it. Pin it to "" instead, and
        # pin the provider so an ambient EMBEDDING_PROVIDER cannot make an
        # unpatched call reach a live service. With both pinned, get_vector_store()
        # returns None rather than opening Chroma.
        cls.original_env = {name: os.environ.get(name) for name in cls.PINNED_ENV}
        os.environ["GOOGLE_API_KEY"] = ""
        os.environ["EMBEDDING_PROVIDER"] = "google"
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

    def test_retrievers_are_created_with_the_callers_user_filter(self):
        get_user_retriever = getattr(self.module, "get_user_retriever", None)
        self.assertIsNotNone(get_user_retriever)

        get_user_retriever("account-a").invoke("question one")
        get_user_retriever("account-b").invoke("question two")

        self.assertEqual(
            self.store.retriever_filters,
            [{"user_id": "account-a"}, {"user_id": "account-b"}],
        )

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

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
    @classmethod
    def setUpClass(cls):
        os.environ.pop("GOOGLE_API_KEY", None)
        sys.modules.pop("langchain_config", None)
        cls.module = importlib.import_module("langchain_config")

    def setUp(self):
        self.original_store = self.module.vector_store
        self.original_embedding_model = self.module.embedding_model
        self.store = FakeVectorStore()
        self.module.vector_store = self.store
        self.module.embedding_model = object()

    def tearDown(self):
        self.module.vector_store = self.original_store
        self.module.embedding_model = self.original_embedding_model

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

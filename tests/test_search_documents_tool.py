import unittest
from unittest.mock import patch

from langchain_core.documents import Document


class FakeVectorStore:
    def __init__(self):
        self.calls = []

    def similarity_search(self, query, k, filter):
        self.calls.append({"query": query, "k": k, "filter": filter})
        return []


class SearchDocumentsToolTests(unittest.TestCase):
    def test_tool_name_is_function_call_safe(self):
        """Gemini and Qwen3 both require ^[a-zA-Z0-9_-]+$. 'RAG Retriever' fails."""
        import langchain_config

        tool = langchain_config.build_rag_tool("user-1")
        self.assertEqual(tool.name, "search_documents")
        self.assertNotIn(" ", tool.name)

    def test_duplicate_chunks_are_collapsed(self):
        import langchain_config

        docs = [
            Document(page_content="same text", metadata={"source": "a.pdf"}),
            Document(page_content="same text", metadata={"source": "a.pdf"}),
            Document(page_content="other text", metadata={"source": "b.pdf"}),
        ]
        with patch.object(langchain_config, "retrieve_for_user", return_value=docs):
            result = langchain_config.build_rag_tool("user-1").invoke(
                {"query": "anything"}
            )

        self.assertEqual(result.count("same text"), 1)

    def test_result_carries_citations(self):
        import langchain_config

        docs = [Document(page_content="body", metadata={"source": "a.pdf"})]
        with patch.object(langchain_config, "retrieve_for_user", return_value=docs):
            result = langchain_config.build_rag_tool("user-1").invoke(
                {"query": "anything"}
            )

        self.assertIn("a.pdf", result)

    def test_empty_retrieval_is_explicit(self):
        """The model must be told retrieval found nothing, so it does not invent."""
        import langchain_config

        with patch.object(langchain_config, "retrieve_for_user", return_value=[]):
            result = langchain_config.build_rag_tool("user-1").invoke(
                {"query": "anything"}
            )

        self.assertIn("No matching", result)

    def test_retrieval_is_scoped_to_the_calling_account(self):
        """The mocked tests above patch out the seam that carries the filter, so
        the filter itself needs its own test. Cross-user leakage is the bug this
        branch exists to prevent."""
        import langchain_config

        store = FakeVectorStore()
        with patch.object(langchain_config, "get_vector_store", return_value=store):
            self.assertEqual(langchain_config.retrieve_for_user("account-a", "q"), [])
            # No user, no retrieval: an unscoped search would read every account.
            self.assertEqual(langchain_config.retrieve_for_user(None, "q"), [])

        with patch.object(langchain_config, "get_vector_store", return_value=None):
            self.assertEqual(langchain_config.retrieve_for_user("account-a", "q"), [])

        self.assertEqual(
            store.calls, [{"query": "q", "k": 4, "filter": {"user_id": "account-a"}}]
        )


if __name__ == "__main__":
    unittest.main()

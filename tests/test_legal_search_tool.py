"""search_legal_corpus: the agent's bridge to the legal_rag statute/case-law service.

legal_rag runs in its own runtime (.venv, Qdrant, reranker), so the Flask agent
reaches it over HTTP. Every test here patches httpx.post; none needs the service.
"""

import itertools
import os
import unittest
from unittest.mock import MagicMock, patch

import httpx


_doc_ids = itertools.count()


def hit(document_type, title, text, score=0.9, doc=None, **metadata):
    """One ScoredChunk; each hit is its own document unless doc= names a shared one."""
    doc = doc if doc is not None else f"d{next(_doc_ids)}"
    return {
        "chunk": {
            "id": f"{doc}#{text[:8]}", "document_id": doc, "document_type": document_type, "path": "p",
            "order": 0, "text": text, "title": title,
            "metadata": {"dataset": "x", "corpus_version": "v1", **metadata},
        },
        "score": score,
        "retriever": "reranked",
    }


def response(hits, labels=None):
    resp = MagicMock()
    resp.json.return_value = {"hits": hits, "labels": labels or {}}
    resp.raise_for_status.return_value = None
    return resp


def call(tool, query="보증금"):
    return tool.invoke({"type": "tool_call", "id": "t1", "name": tool.name, "args": {"query": query}})


class LegalSearchToolTests(unittest.TestCase):
    def setUp(self):
        import langchain_config
        self.lc = langchain_config
        self.tool = langchain_config.build_legal_search_tool("http://legal:8000/")

    def test_posts_the_query_to_the_search_endpoint(self):
        with patch.object(self.lc.httpx, "post", return_value=response([])) as post:
            call(self.tool, "임차권등기명령")

        url = post.call_args.args[0]
        self.assertEqual(url, "http://legal:8000/v1/search")
        self.assertEqual(post.call_args.kwargs["json"]["query"], "임차권등기명령")

    def test_hits_become_labelled_evidence(self):
        hits = [
            hit("judgment", "임대차보증금", "원고는 보증금 반환을 구한다", 0.96, court="서울서부지방법원"),
            hit("statute", "주택임대차보호법", "임차권등기명령을 신청할 수 있다", 0.9,
                law_name="주택임대차보호법", article="제3조의3"),
            hit("legal_interpretation", "법령해석", "해석 본문", 0.8),
            hit("patent_trial_decision", "특허심판", "심결 본문", 0.7),
        ]
        with patch.object(self.lc.httpx, "post", return_value=response(hits)):
            message = call(self.tool)

        kinds = [h["kind"] for h in message.artifact]
        self.assertEqual(kinds, ["판례", "법령", "해석례", "판례"])
        self.assertEqual(message.artifact[0]["source"], "서울서부지방법원 임대차보증금")
        self.assertEqual(message.artifact[1]["source"], "주택임대차보호법 제3조의3")
        self.assertEqual(message.artifact[0]["score"], 0.96)
        # The model reads the prose, so the kind and source must be in it too.
        self.assertIn("[판례] 서울서부지방법원 임대차보증금", message.content)
        self.assertIn("임차권등기명령을 신청할 수 있다", message.content)

    def test_service_label_names_an_otherwise_anonymous_hit(self):
        """legal_rag labels middle chunks from their document's header (labels by document_id)."""
        hits = [hit("judgment", "", "피고인은 피해자를 기망하여", doc="d")]
        labels = {"d": "광주지방법원 2022고단4222 사기"}
        with patch.object(self.lc.httpx, "post", return_value=response(hits, labels)):
            message = call(self.tool)

        self.assertEqual(message.artifact[0]["source"], "광주지방법원 2022고단4222 사기")
        self.assertIn("[판례] 광주지방법원 2022고단4222 사기", message.content)

    def test_results_end_with_a_korean_answer_reminder(self):
        """qwen3:4b drifted into English right after a tool result (2026-09-27);
        the last thing it reads before answering is the tool output."""
        hits = [hit("judgment", "임대차보증금", "본문", court="서울서부지방법원")]
        with patch.object(self.lc.httpx, "post", return_value=response(hits)):
            message = call(self.tool)

        self.assertTrue(message.content.rstrip().endswith(self.lc.LEGAL_ANSWER_LANGUAGE_REMINDER))
        self.assertIn("한국어", self.lc.LEGAL_ANSWER_LANGUAGE_REMINDER)

    def test_untitled_hit_falls_back_to_its_first_line(self):
        hits = [hit("constitutional_decision", "", "사 건 2018헌바434 위헌소원\n본문")]
        with patch.object(self.lc.httpx, "post", return_value=response(hits)):
            message = call(self.tool)

        self.assertEqual(message.artifact[0]["source"], "사 건 2018헌바434 위헌소원")

    def test_no_hits_is_explicit(self):
        with patch.object(self.lc.httpx, "post", return_value=response([])):
            message = call(self.tool)

        self.assertIn("No matching", message.content)
        self.assertEqual(message.artifact, [])

    def test_unreachable_service_is_reported_not_raised(self):
        """A down legal_rag must not kill the chat turn; the model has to say so."""
        with patch.object(self.lc.httpx, "post", side_effect=httpx.ConnectError("refused")):
            message = call(self.tool)

        self.assertIn("unavailable", message.content)
        self.assertEqual(message.artifact, [])


class LegalSearchCollapseTests(unittest.TestCase):
    """Several chunks of one document are one piece of evidence, not repeated sources."""

    def setUp(self):
        import langchain_config
        self.lc = langchain_config
        self.tool = langchain_config.build_legal_search_tool("http://legal:8000/")

    def run_tool(self, hits, labels=None):
        with patch.object(self.lc.httpx, "post", return_value=response(hits, labels)) as post:
            message = call(self.tool)
        return message, post

    def test_chunks_of_one_document_collapse_into_one_entry(self):
        hits = [
            hit("constitutional_decision", "", "첫째 조각", 0.97, doc="2011헌마233"),
            hit("statute", "주택임대차보호법", "법령 본문", 0.95, doc="law"),
            hit("constitutional_decision", "", "둘째 조각", 0.93, doc="2011헌마233"),
            hit("constitutional_decision", "", "셋째 조각", 0.90, doc="2011헌마233"),
        ]
        labels = {"2011헌마233": "헌법재판소 2011헌마233"}
        message, _ = self.run_tool(hits, labels)

        self.assertEqual([e["source"] for e in message.artifact],
                         ["헌법재판소 2011헌마233", "주택임대차보호법"])
        self.assertEqual(message.content.count("헌법재판소 2011헌마233"), 1)

    def test_collapsed_entry_keeps_rank_order_and_best_score(self):
        hits = [
            hit("judgment", "A", "a1", 0.99, doc="A"),
            hit("judgment", "B", "b1", 0.95, doc="B"),
            hit("judgment", "A", "a2", 0.50, doc="A"),
        ]
        message, _ = self.run_tool(hits)

        self.assertEqual([e["source"] for e in message.artifact], ["A", "B"])
        self.assertEqual([e["score"] for e in message.artifact], [0.99, 0.95])

    def test_collapsed_quote_joins_distinct_chunk_texts(self):
        hits = [
            hit("judgment", "A", "첫째 조각", 0.9, doc="A"),
            hit("judgment", "A", "둘째 조각", 0.8, doc="A"),
            hit("judgment", "A", "첫째 조각", 0.7, doc="A"),  # the same text twice
        ]
        message, _ = self.run_tool(hits)

        self.assertEqual(len(message.artifact), 1)
        self.assertEqual(message.artifact[0]["quote"], "첫째 조각\n…\n둘째 조각")

    def test_collapsed_quote_is_capped_per_document(self):
        cap = self.lc.LEGAL_QUOTE_CHARS
        hits = [
            hit("judgment", "A", "가" * cap, 0.9, doc="A"),
            hit("judgment", "A", "나" * cap, 0.8, doc="A"),
            hit("judgment", "B", "다" * cap, 0.7, doc="B"),
        ]
        message, _ = self.run_tool(hits)

        self.assertEqual([len(e["quote"]) for e in message.artifact], [cap, cap])
        self.assertTrue(message.artifact[0]["quote"].startswith("가"))

    def test_asks_the_service_for_extra_hits_to_survive_collapsing(self):
        _, post = self.run_tool([])

        k = post.call_args.kwargs["json"]["k"]
        self.assertEqual(k, self.lc.LEGAL_SEARCH_K)
        self.assertGreater(k, self.lc.LEGAL_MAX_DOCUMENTS)
        self.assertLessEqual(k, 50)  # the service's cap

    def test_keeps_at_most_the_document_limit(self):
        limit = self.lc.LEGAL_MAX_DOCUMENTS
        hits = [hit("judgment", f"사건{i}", f"본문{i}", 1 - i / 100, doc=f"doc{i}") for i in range(limit + 3)]
        message, _ = self.run_tool(hits)

        self.assertEqual([e["source"] for e in message.artifact], [f"사건{i}" for i in range(limit)])

    def test_hits_without_a_document_id_are_not_merged(self):
        hits = [hit("judgment", "A", "a1", doc=""), hit("judgment", "B", "b1", doc="")]
        for h in hits:
            h["chunk"].pop("document_id")
        message, _ = self.run_tool(hits)

        self.assertEqual([e["source"] for e in message.artifact], ["A", "B"])


class LegalSearchWiringTests(unittest.TestCase):
    class StubModel:
        def bind_tools(self, tools, **kwargs):
            return self

    def tool_names(self, env):
        import langchain_config

        with patch.dict(os.environ, env, clear=False):
            if not env.get("LEGAL_RAG_URL"):
                os.environ.pop("LEGAL_RAG_URL", None)
            with patch.object(langchain_config, "get_chat_model", return_value=self.StubModel()), \
                 patch("langchain.agents.create_agent") as create:
                langchain_config.build_agent("user-1")
        return [t.name for t in create.call_args.kwargs["tools"]]

    def test_tool_is_off_without_a_configured_service(self):
        self.assertEqual(self.tool_names({}), ["search_documents"])

    def test_tool_is_registered_when_the_service_is_configured(self):
        self.assertEqual(
            self.tool_names({"LEGAL_RAG_URL": "http://127.0.0.1:8000"}),
            ["search_documents", "search_legal_corpus"],
        )


if __name__ == "__main__":
    unittest.main()

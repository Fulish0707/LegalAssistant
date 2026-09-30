"""The server searches statutes and case law before the model answers.

A prompt rule was not enough: qwen3:4b skipped search_legal_corpus on "해고당한 지 두 달
됐는데 지금이라도 구제신청할 수 있나요?" with its full instructions in view and answered
"30일, 제115조" (it is 3개월, 제28조 제2항). The evidence rides in the turn's system prompt
through middleware, so it is never checkpointed into the thread.
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from tests.test_chat_streaming import SCHEMA_JSON, ChatStreamBase, StubAgent

HIT = {"source": "헌법재판소 2011헌마233", "quote": "3개월 이내", "kind": "판례", "score": 0.9}
EVIDENCE = "[판례] 헌법재판소 2011헌마233\n3개월 이내\n\n(위 근거를 바탕으로 사용자에게 반드시 한국어로 답하세요.)"


def final_agent():
    return StubAgent([("values", {"messages": [AIMessage(content=SCHEMA_JSON)]})])


class ForcedRetrievalStreamTests(ChatStreamBase):
    def run_turn(self, message, url="http://legal:8000", search=None):
        agent = final_agent()
        search = search or mock.Mock(return_value=(EVIDENCE, [HIT]))
        env = {"LEGAL_RAG_URL": url} if url else {}
        with mock.patch.dict(os.environ, env), mock.patch.object(self.app_module, "search_legal_evidence", search):
            if not url:
                os.environ.pop("LEGAL_RAG_URL", None)
            events = self.collect(agent, "message=" + message)
        final = [e for e in events if e["type"] == "final_response"][0]
        return agent.stream_calls[0][2]["context"], final, search, events

    def test_legal_question_is_searched_before_the_model_runs(self):
        context, final, search, _ = self.run_turn("해고당한 지 두 달 됐는데 구제신청할 수 있나요?")

        search.assert_called_once_with("http://legal:8000", "해고당한 지 두 달 됐는데 구제신청할 수 있나요?")
        self.assertEqual(context.legal_evidence, EVIDENCE)
        self.assertEqual(final["sources"], [HIT])

    def test_greeting_is_not_searched(self):
        context, final, search, _ = self.run_turn("안녕하세요")

        search.assert_not_called()
        self.assertEqual(context.legal_evidence, "")
        self.assertEqual(final["sources"], [])

    def test_no_search_without_a_configured_service(self):
        context, _, search, _ = self.run_turn("보증금을 못 받았어요", url=None)

        search.assert_not_called()
        self.assertEqual(context.legal_evidence, "")

    def test_unavailable_search_still_answers_and_tells_the_model(self):
        down = mock.Mock(return_value=("The statute and case-law search is unavailable right now (ConnectError).", []))
        context, final, _, events = self.run_turn("보증금을 못 받았어요", search=down)

        self.assertIn("unavailable", context.legal_evidence)
        self.assertEqual(final["sources"], [])
        self.assertEqual(events[-1]["type"], "stream_end")


class LegalEvidenceMiddlewareTests(unittest.TestCase):
    def request(self, evidence):
        import langchain_config
        from langchain.agents.middleware import ModelRequest

        @tool("search_documents")
        def docs(query: str) -> str:
            """d"""
            return ""

        @tool("search_legal_corpus")
        def legal(query: str) -> str:
            """l"""
            return ""

        runtime = SimpleNamespace(context=langchain_config.AgentContext(user_id="u", legal_evidence=evidence))
        return ModelRequest(model=mock.Mock(), messages=[], system_prompt="BASE", tools=[docs, legal], runtime=runtime)

    def apply(self, request):
        import langchain_config

        return langchain_config.legal_evidence_middleware.wrap_model_call(request, lambda r: r)

    def test_evidence_is_appended_to_the_system_prompt_and_the_search_tool_hidden(self):
        seen = self.apply(self.request(EVIDENCE))

        self.assertTrue(seen.system_prompt.startswith("BASE"))
        self.assertIn(EVIDENCE, seen.system_prompt)
        self.assertEqual([t.name for t in seen.tools], ["search_documents"])

    def test_without_evidence_the_request_is_unchanged(self):
        seen = self.apply(self.request(""))

        self.assertEqual(seen.system_prompt, "BASE")
        self.assertEqual([t.name for t in seen.tools], ["search_documents", "search_legal_corpus"])


if __name__ == "__main__":
    unittest.main()

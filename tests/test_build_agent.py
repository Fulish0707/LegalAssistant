import unittest
from unittest.mock import patch


class StubModel:
    """Stands in for a provider model that can run a tool loop.

    create_agent only needs bind_tools to exist to compile the graph, so this
    keeps the test off the network without weakening what is asserted.
    """

    def bind_tools(self, tools, **kwargs):
        return self


class BuildAgentTests(unittest.TestCase):
    def test_returns_none_when_no_model_configured(self):
        import langchain_config

        with patch.object(langchain_config, "get_chat_model", return_value=None):
            self.assertIsNone(langchain_config.build_agent("user-1"))

    def test_model_without_bind_tools_fails_loudly(self):
        """create_agent requires bind_tools. qwen3:4b has it (verified), so a model
        that lacks it means a misconfigured provider -- say so, do not silently
        degrade into a different product."""
        import langchain_config

        class NoToolModel:
            pass

        with patch.object(langchain_config, "get_chat_model", return_value=NoToolModel()):
            with self.assertRaises(RuntimeError) as caught:
                langchain_config.build_agent("user-1")

        self.assertIn("bind_tools", str(caught.exception))

    def test_rag_tool_absent_for_anonymous_callers(self):
        import langchain_config

        with patch.object(langchain_config, "get_chat_model", return_value=StubModel()):
            with patch.object(langchain_config, "build_rag_tool") as rag:
                langchain_config.build_agent(None)
                rag.assert_not_called()

    def test_rag_tool_present_for_signed_in_callers(self):
        import langchain_config

        real_tool = langchain_config.build_rag_tool("user-1")
        with patch.object(langchain_config, "get_chat_model", return_value=StubModel()):
            with patch.object(langchain_config, "build_rag_tool", return_value=real_tool) as rag:
                langchain_config.build_agent("user-1")
                rag.assert_called_once_with("user-1")

    def test_search_tools_are_built_lazily_not_at_import(self):
        import langchain_config

        self.assertTrue(callable(langchain_config.get_search_tools))
        self.assertFalse(hasattr(langchain_config, "serpapi_tools"))


class ExtractAgentSectionsTests(unittest.TestCase):
    """The four sections must survive both provider shapes.

    The JSON literal below is copied verbatim from a live qwen3:4b run.
    """

    QWEN_CONTENT = (
        '{\n  "chat_response": "Hello! I\'m PS_Agent, your practical AI work '
        'companion for university applications, professional reports, and more.",\n'
        '  "statement_suggestion": "",\n  "job_statement_suggestion": "",\n'
        '  "report_suggestion": ""\n}'
    )

    def test_reads_json_written_into_the_final_message_content(self):
        import langchain_config
        from langchain_core.messages import AIMessage, HumanMessage

        sections = langchain_config.extract_agent_sections({
            "structured_response": None,
            "messages": [HumanMessage("hi"), AIMessage(self.QWEN_CONTENT)],
        })

        self.assertIsNotNone(sections)
        self.assertIn("PS_Agent", sections["chat_response"])
        self.assertNotIn("chat_response", sections["chat_response"])
        self.assertEqual(sections["statement_suggestion"], "")
        self.assertEqual(sections["report_suggestion"], "")

    def test_reads_a_structured_response_model_instance(self):
        import langchain_config

        sections = langchain_config.extract_agent_sections({
            "structured_response": langchain_config.AgentSections(
                chat_response="Here is the answer.",
                report_suggestion="Tighten the summary.",
            ),
            "messages": [],
        })

        self.assertEqual(sections["chat_response"], "Here is the answer.")
        self.assertEqual(sections["report_suggestion"], "Tighten the summary.")
        self.assertEqual(sections["job_statement_suggestion"], "")

    def test_unwraps_a_json_code_fence(self):
        import langchain_config
        from langchain_core.messages import AIMessage

        sections = langchain_config.extract_agent_sections({
            "messages": [AIMessage("```json\n" + self.QWEN_CONTENT + "\n```")],
        })

        self.assertIn("PS_Agent", sections["chat_response"])

    def test_returns_none_for_legacy_marker_prose_so_the_caller_can_fall_back(self):
        import langchain_config
        from langchain_core.messages import AIMessage

        self.assertIsNone(langchain_config.extract_agent_sections({
            "messages": [AIMessage("[CHAT] Hello there.")],
        }))
        self.assertIsNone(langchain_config.extract_agent_sections({"messages": []}))
        self.assertIsNone(langchain_config.extract_agent_sections(None))


if __name__ == "__main__":
    unittest.main()

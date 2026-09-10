"""SSE contract tests for /chat.

The subject here is the event contract static/lc-chat.js consumes, not model
output: every test patches build_agent with a stub whose .stream() yields fixed
LangGraph chunks. A real agent would run live inference on every suite run.
"""

import importlib
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage


# Exactly the set static/lc-chat.js branches on (sse.onmessage, lines 266-282).
FRONTEND_EVENT_TYPES = {
    "token_chunk",
    "thought",
    "final_response",
    "update_draft",
    "error",
    "stream_end",
}

SCHEMA_JSON = json.dumps(
    {
        "chat_response": "보증금은 돌려받을 수 있습니다.",
        "statement_suggestion": "",
        "job_statement_suggestion": "",
        "report_suggestion": "",
    },
    ensure_ascii=False,
)


class StubAgent:
    """Stands in for a compiled LangGraph agent."""

    def __init__(self, events, state_messages=None):
        self._events = events
        self._state_messages = state_messages
        self.stream_calls = []

    def stream(self, payload, config=None, **kwargs):
        self.stream_calls.append((payload, config, kwargs))
        yield from self._events

    def get_state(self, config):
        if self._state_messages is None:
            raise ValueError("No checkpointer set")
        return mock.Mock(values={"messages": self._state_messages})


def ai_chunks(text, message_id="run-1"):
    """One ("messages", ...) event per character-run, as the model streams."""
    return [
        ("messages", (AIMessageChunk(content=piece, id=message_id), {}))
        for piece in text
    ]


class ChatStreamBase(unittest.TestCase):
    MODULE_NAMES = ("app", "langchain_config", "llm_model", "llm_prompt")

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.saved_env = {
            name: os.environ.get(name)
            for name in ("APP_STATE_DB", "AGENT_CHECKPOINT_DB", "RAG_PERSIST_DIR", "APP_ENV", "LANGSMITH_TRACING")
        }
        cls.saved_modules = {name: sys.modules.get(name) for name in cls.MODULE_NAMES}
        os.environ["APP_STATE_DB"] = os.path.join(cls.tmpdir.name, "app_state.db")
        os.environ["AGENT_CHECKPOINT_DB"] = os.path.join(cls.tmpdir.name, "checkpoints.db")
        os.environ["RAG_PERSIST_DIR"] = os.path.join(cls.tmpdir.name, "chroma")
        os.environ["APP_ENV"] = "testing"
        os.environ["LANGSMITH_TRACING"] = "false"
        for name in cls.MODULE_NAMES:
            sys.modules.pop(name, None)
        cls.app_module = importlib.import_module("app")
        cls.client = cls.app_module.app.test_client()

    @classmethod
    def tearDownClass(cls):
        for name in cls.MODULE_NAMES:
            sys.modules.pop(name, None)
            if cls.saved_modules[name] is not None:
                sys.modules[name] = cls.saved_modules[name]
        for name, value in cls.saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        cls.tmpdir.cleanup()

    def as_browser(self, user_id):
        """Pin the class client's workspace id, the way a returning browser has one."""
        with self.client.session_transaction() as flask_session:
            flask_session["user_id"] = user_id

    def collect(self, agent, query="message=hello"):
        with mock.patch.object(self.app_module, "build_agent", return_value=agent):
            response = self.client.get("/chat?" + query)
            body = response.get_data(as_text=True)
        return [
            json.loads(line[len("data: "):])
            for line in body.splitlines()
            if line.startswith("data: ")
        ]


class ChatStreamingTests(ChatStreamBase):
    def test_stream_emits_known_event_types_and_terminates(self):
        agent = StubAgent(
            ai_chunks("Hello there.")
            + [("values", {"messages": [AIMessage(content=SCHEMA_JSON)]})]
        )
        events = self.collect(agent)

        self.assertTrue(events)
        for event in events:
            self.assertIn(event["type"], FRONTEND_EVENT_TYPES)
        self.assertEqual(events[-1]["type"], "stream_end")
        self.assertEqual(events[-2]["type"], "final_response")

    def test_final_response_carries_the_four_sections(self):
        agent = StubAgent(
            [("values", {"messages": [AIMessage(content=SCHEMA_JSON)]})]
        )
        final = [e for e in self.collect(agent) if e["type"] == "final_response"][0]

        self.assertEqual(final["chat_response"], "보증금은 돌려받을 수 있습니다.")
        for field in ("statement_suggestion", "job_statement_suggestion", "report_suggestion"):
            self.assertIn(field, final)

    def test_schema_json_never_leaks_into_token_chunks(self):
        """qwen3:4b writes the schema JSON as the final message content."""
        agent = StubAgent(
            ai_chunks(SCHEMA_JSON)
            + [("values", {"messages": [AIMessage(content=SCHEMA_JSON)]})]
        )
        events = self.collect(agent)

        streamed = "".join(e["content"] for e in events if e["type"] == "token_chunk")
        self.assertNotIn("job_statement_suggestion", streamed)
        self.assertNotIn("chat_response", streamed)
        self.assertNotIn("{", streamed)
        final = [e for e in events if e["type"] == "final_response"][0]
        self.assertEqual(final["chat_response"], "보증금은 돌려받을 수 있습니다.")

    def test_reasoning_blocks_are_not_streamed_as_answer_tokens(self):
        agent = StubAgent(
            ai_chunks("<think>먼저 검색한다</think>답변입니다.")
            + [("values", {"messages": [AIMessage(content='{"chat_response": "답변입니다."}')]})]
        )
        streamed = "".join(
            e["content"] for e in self.collect(agent) if e["type"] == "token_chunk"
        )
        self.assertNotIn("먼저 검색한다", streamed)
        self.assertNotIn("<think>", streamed)
        self.assertEqual(streamed.strip(), "답변입니다.")

    def test_tool_messages_do_not_stream_as_answer_tokens(self):
        agent = StubAgent(
            [("messages", (ToolMessage(content="RAG hit: 임대차보호법 제3조", tool_call_id="t1"), {}))]
            + ai_chunks("정리하면 이렇습니다.")
            + [("values", {"messages": [AIMessage(content="정리하면 이렇습니다.")]})]
        )
        events = self.collect(agent)
        streamed = "".join(e["content"] for e in events if e["type"] == "token_chunk")

        self.assertNotIn("RAG hit", streamed)
        self.assertEqual(streamed.strip(), "정리하면 이렇습니다.")

    def test_tool_loop_does_not_truncate_the_final_turn(self):
        """on_llm_end used to break the drain loop after the first model call."""
        agent = StubAgent(
            ai_chunks("먼저 찾아보겠습니다. ", message_id="run-1")
            + [("updates", {"tools": {"messages": [ToolMessage(content="근거", tool_call_id="t1")]}})]
            + ai_chunks("최종 답변입니다.", message_id="run-2")
            + [("values", {"messages": [AIMessage(content="최종 답변입니다.")]})]
        )
        events = self.collect(agent)
        streamed = "".join(e["content"] for e in events if e["type"] == "token_chunk")

        self.assertIn("최종 답변입니다.", streamed)
        self.assertTrue([e for e in events if e["type"] == "thought"])

    def test_stream_is_threaded_on_the_browsers_workspace_and_terminates(self):
        agent = StubAgent(
            ai_chunks("안녕하세요.")
            + [("values", {"messages": [AIMessage(content="안녕하세요.")]})]
        )
        events = self.collect(agent)

        self.assertEqual(events[-1]["type"], "stream_end")
        _payload, config, kwargs = agent.stream_calls[0]
        with self.client.session_transaction() as flask_session:
            browser_id = flask_session["user_id"]
        self.assertEqual(config["configurable"]["thread_id"], browser_id)
        self.assertEqual(kwargs["context"].user_id, browser_id)

    def test_missing_message_still_ends_the_stream(self):
        response = self.client.get("/chat")
        body = response.get_data(as_text=True)
        types = [
            json.loads(line[len("data: "):])["type"]
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual(types, ["error", "stream_end"])


class DraftContextTests(ChatStreamBase):
    """The draft must not accumulate into every turn of a persisted thread."""

    def test_a_first_turn_carries_the_draft(self):
        """StubAgent.get_state raises when no thread state exists: a new
        workspace's first turn, where nothing has carried the draft yet."""
        agent = StubAgent([("values", {"messages": [AIMessage(content="ok")]})])
        self.collect(agent, "message=hi&personal_statement_draft=MY_DRAFT")
        payload, _config, _kwargs = agent.stream_calls[0]
        self.assertIn("MY_DRAFT", payload["messages"][0][1])

    def test_unchanged_draft_is_not_resent_when_the_thread_already_has_it(self):
        prior = HumanMessage(content="hi\n\n[Current Draft Context]\nMY_DRAFT")
        agent = StubAgent(
            [("values", {"messages": [AIMessage(content="ok")]})],
            state_messages=[prior, AIMessage(content="ok")],
        )
        self.as_browser("u1")
        self.collect(agent, "message=hi+again&personal_statement_draft=MY_DRAFT")
        payload, config, _kwargs = agent.stream_calls[0]
        self.assertNotIn("MY_DRAFT", payload["messages"][0][1])
        self.assertEqual(config["configurable"]["thread_id"], "u1")

    def test_unchanged_draft_stays_out_of_later_turns(self):
        """The naive check -- look only at the newest user turn -- re-sends the
        draft on every other turn, because the turn that skipped it has none."""
        agent = StubAgent(
            [("values", {"messages": [AIMessage(content="ok")]})],
            state_messages=[
                HumanMessage(content="turn one\n\n[Current Draft Context]\nMY_DRAFT"),
                AIMessage(content="ok"),
                HumanMessage(content="turn two"),
                AIMessage(content="ok"),
            ],
        )
        self.as_browser("u3")
        self.collect(agent, "message=turn+three&personal_statement_draft=MY_DRAFT")
        payload, _config, _kwargs = agent.stream_calls[0]
        self.assertNotIn("MY_DRAFT", payload["messages"][0][1])

    def test_changed_draft_is_resent(self):
        prior = HumanMessage(content="hi\n\n[Current Draft Context]\nOLD_DRAFT")
        agent = StubAgent(
            [("values", {"messages": [AIMessage(content="ok")]})],
            state_messages=[prior, AIMessage(content="ok")],
        )
        self.as_browser("u2")
        self.collect(agent, "message=hi+again&personal_statement_draft=NEW_DRAFT")
        payload, _config, _kwargs = agent.stream_calls[0]
        self.assertIn("NEW_DRAFT", payload["messages"][0][1])


class AnswerTokenFilterTests(ChatStreamBase):
    """Unit tests for the decide-once gate that keeps schema JSON off the wire."""

    @property
    def filter_cls(self):
        return self.app_module._AnswerTokenFilter

    def feed_all(self, pieces):
        f = self.filter_cls()
        out = "".join(f.feed(p) for p in pieces)
        return out + f.flush()

    def test_prose_passes_through(self):
        self.assertEqual(self.feed_all(["Hel", "lo ", "world"]), "Hello world")

    def test_json_object_is_suppressed_entirely(self):
        self.assertEqual(self.feed_all(['{"chat_', 'response": "hi"}']), "")

    def test_fenced_json_is_suppressed(self):
        self.assertEqual(self.feed_all(["```json\n", '{"a": 1}\n```']), "")

    def test_think_block_split_across_chunks_is_stripped(self):
        self.assertEqual(self.feed_all(["<thi", "nk>secret</th", "ink>answer"]), "answer")

    def test_json_after_a_think_block_is_still_suppressed(self):
        self.assertEqual(self.feed_all(["<think>plan</think>", '{"chat_response": "x"}']), "")

    def test_list_content_keeps_only_text_blocks(self):
        text = self.app_module._message_text(
            [{"type": "reasoning", "reasoning": "hmm"}, {"type": "text", "text": "hi"}]
        )
        self.assertEqual(text, "hi")


if __name__ == "__main__":
    unittest.main()

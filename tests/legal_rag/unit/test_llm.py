import json

import httpx
import pytest
from pydantic import BaseModel

from legal_rag.llm import FakeChatModel, Message, OllamaChatModel, StructuredOutputError
from legal_rag.llm.prompts import load_prompt


class Out(BaseModel):
    answer: str


def test_fake_model_returns_scripted_structured_responses_in_order():
    m = FakeChatModel({"Out": [Out(answer="first"), '{"answer": "second"}'], "text": ["hi"]})
    assert m.generate([Message(role="user", content="x")], response_schema=Out).parsed.answer == "first"
    assert m.generate([Message(role="user", content="x")], response_schema=Out).parsed.answer == "second"
    assert m.generate([Message(role="user", content="x")]).text == "hi"
    assert len(m.calls) == 3 and m.calls[0].response_schema is Out
    with pytest.raises(AssertionError):
        m.generate([Message(role="user", content="x")], response_schema=Out)


def test_ollama_model_sends_schema_and_parses():
    seen = {}

    def handler(request: httpx.Request):
        seen.update(json.loads(request.read()))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": '```json\n{"answer": "ok"}\n```'}})

    m = OllamaChatModel(model="qwen3:4b", base_url="http://t", transport=httpx.MockTransport(handler))
    r = m.generate([Message(role="user", content="q")], response_schema=Out, temperature=0.0, max_tokens=99)
    assert r.parsed.answer == "ok" and m.name == "ollama:qwen3:4b"
    assert seen["format"]["type"] == "object" and seen["think"] is False and seen["stream"] is False
    assert seen["options"] == {"temperature": 0.0, "num_predict": 99}


def test_ollama_num_ctx_in_options_only_when_set():
    seen = []

    def handler(request: httpx.Request):
        seen.append(json.loads(request.read()))
        return httpx.Response(200, json={"message": {"content": "x"}})

    OllamaChatModel(model="m", base_url="http://t", transport=httpx.MockTransport(handler), num_ctx=8192).generate([Message(role="user", content="q")])
    OllamaChatModel(model="m", base_url="http://t", transport=httpx.MockTransport(handler)).generate([Message(role="user", content="q")])
    assert seen[0]["options"]["num_ctx"] == 8192
    assert "num_ctx" not in seen[1]["options"]


def test_ollama_plain_text_without_schema():
    def handler(req):
        return httpx.Response(200, json={"message": {"content": "안녕"}})
    m = OllamaChatModel(model="m", base_url="http://t", transport=httpx.MockTransport(handler))
    assert m.generate([Message(role="user", content="q")]).text == "안녕"


def test_ollama_invalid_json_raises_structured_output_error():
    def handler(req):
        return httpx.Response(200, json={"message": {"content": "not json"}})
    m = OllamaChatModel(model="m", base_url="http://t", transport=httpx.MockTransport(handler))
    with pytest.raises(StructuredOutputError) as info:
        m.generate([Message(role="user", content="q")], response_schema=Out)
    assert info.value.text == "not json"


def test_prompts_load():
    assert "LEGAL SOURCE" in load_prompt("answer_system")
    assert "needs_more_info" in load_prompt("analyze_system")
    assert "search_queries" in load_prompt("rewrite_system")


def test_ollama_streams_content_to_the_callback_and_parses_the_whole():
    seen, pieces = {}, []
    lines = [{"message": {"content": '{"answer": '}}, {"message": {"content": '"ok"}'}},
             {"message": {"content": ""}, "done": True, "eval_count": 2}]

    def handler(request: httpx.Request):
        seen.update(json.loads(request.read()))
        return httpx.Response(200, content="".join(json.dumps(line) + "\n" for line in lines))

    m = OllamaChatModel(model="m", base_url="http://t", transport=httpx.MockTransport(handler))
    r = m.generate([Message(role="user", content="q")], response_schema=Out, on_text=pieces.append)
    assert seen["stream"] is True and pieces == ['{"answer": ', '"ok"}']
    assert r.parsed.answer == "ok" and r.raw["eval_count"] == 2


def test_ollama_stream_error_line_raises():
    def handler(request: httpx.Request):
        return httpx.Response(200, content=json.dumps({"error": "model crashed"}) + "\n")

    m = OllamaChatModel(model="m", base_url="http://t", transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="model crashed"):
        m.generate([Message(role="user", content="q")], response_schema=Out, on_text=lambda piece: None)


def test_fake_model_passes_its_text_to_the_callback():
    pieces = []
    FakeChatModel({"Out": ['{"answer": "x"}']}).generate([Message(role="user", content="q")], response_schema=Out, on_text=pieces.append)
    assert pieces == ['{"answer": "x"}']

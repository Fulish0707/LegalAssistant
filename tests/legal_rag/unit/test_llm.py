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


def test_ollama_plain_text_without_schema():
    handler = lambda req: httpx.Response(200, json={"message": {"content": "안녕"}})
    m = OllamaChatModel(model="m", base_url="http://t", transport=httpx.MockTransport(handler))
    assert m.generate([Message(role="user", content="q")]).text == "안녕"


def test_ollama_invalid_json_raises_structured_output_error():
    handler = lambda req: httpx.Response(200, json={"message": {"content": "not json"}})
    m = OllamaChatModel(model="m", base_url="http://t", transport=httpx.MockTransport(handler))
    with pytest.raises(StructuredOutputError) as info:
        m.generate([Message(role="user", content="q")], response_schema=Out)
    assert info.value.text == "not json"


def test_prompts_load():
    assert "LEGAL SOURCE" in load_prompt("answer_system")
    assert "needs_more_info" in load_prompt("analyze_system")
    assert "search_queries" in load_prompt("rewrite_system")

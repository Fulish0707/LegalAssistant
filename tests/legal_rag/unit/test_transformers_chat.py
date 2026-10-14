"""Tests for TransformersChatModel — uses a monkeypatched model to avoid GPU."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from legal_rag.llm.base import Message, StructuredOutputError


class SimpleAnswer(BaseModel):
    summary: str
    confidence: str


# ── helpers ─────────────────────────────────────────────────────────────────


def _fake_tokenizer():
    tok = MagicMock()
    tok.apply_chat_template = MagicMock(return_value="<prompt>")
    tok.__call__ = MagicMock(
        return_value=MagicMock(
            to=MagicMock(return_value={"input_ids": MagicMock(shape=[1, 10])}),
        )
    )

    def _decode(ids, skip_special_tokens=True):
        return ids  # we'll set this via generate return

    tok.decode = _decode
    return tok


def _make_model(output_text: str, adapter: str | None = None):
    """Create a TransformersChatModel with mocked internals."""
    from legal_rag.llm.transformers import TransformersChatModel

    model = TransformersChatModel(model_id="test/model", adapter=adapter)

    # Pretend we're loaded
    mock_inner = MagicMock()
    mock_inner.device = "cpu"

    import torch

    fake_output = torch.tensor([[0] * 10])  # prompt tokens
    mock_inner.generate = MagicMock(return_value=fake_output)

    model._model = mock_inner
    tok = MagicMock()
    tok.apply_chat_template = MagicMock(return_value="<prompt>")

    input_ids = torch.tensor([[0] * 10])
    tok_result = MagicMock()
    tok_result.__getitem__ = lambda self, key: input_ids
    tok_result.to = MagicMock(return_value={"input_ids": input_ids})
    tok.__call__ = MagicMock(return_value=tok_result)
    tok.decode = MagicMock(return_value=output_text)

    model._tokenizer = tok
    return model


# ── tests ───────────────────────────────────────────────────────────────────


def test_plain_text_generation():
    model = _make_model("법률에 따르면 제10조에 해당합니다.")
    messages = [Message(role="user", content="질문입니다")]
    resp = model.generate(messages, temperature=0)
    assert resp.text == "법률에 따르면 제10조에 해당합니다."
    assert resp.parsed is None


def test_structured_output_success():
    json_text = '{"summary": "요약", "confidence": "high"}'
    model = _make_model(json_text)
    messages = [Message(role="user", content="질문")]
    resp = model.generate(messages, response_schema=SimpleAnswer, temperature=0)
    assert resp.parsed is not None
    assert resp.parsed.summary == "요약"
    assert resp.parsed.confidence == "high"


def test_structured_output_with_fence():
    fenced = '```json\n{"summary": "결론", "confidence": "medium"}\n```'
    model = _make_model(fenced)
    messages = [Message(role="user", content="질문")]
    resp = model.generate(messages, response_schema=SimpleAnswer, temperature=0)
    assert resp.parsed.summary == "결론"


def test_structured_output_with_surrounding_text():
    text = '답변입니다.\n{"summary": "핵심", "confidence": "low"}\n이상입니다.'
    model = _make_model(text)
    messages = [Message(role="user", content="질문")]
    resp = model.generate(messages, response_schema=SimpleAnswer, temperature=0)
    assert resp.parsed.summary == "핵심"


def test_structured_output_failure():
    model = _make_model("이건 JSON이 아닙니다.")
    messages = [Message(role="user", content="질문")]
    with pytest.raises(StructuredOutputError):
        model.generate(messages, response_schema=SimpleAnswer, temperature=0)


def test_name_includes_adapter():
    from legal_rag.llm.transformers import TransformersChatModel

    base = TransformersChatModel(model_id="Qwen/Qwen3-4B")
    assert "adapter" not in base.name

    adapted = TransformersChatModel(model_id="Qwen/Qwen3-4B", adapter="/path/to/adapter")
    assert "+adapter" in adapted.name


def test_greedy_when_temperature_zero():
    model = _make_model("답변")
    messages = [Message(role="user", content="질문")]
    model.generate(messages, temperature=0)
    call_kwargs = model._model.generate.call_args
    assert call_kwargs.kwargs.get("do_sample", call_kwargs[1].get("do_sample")) is False


def test_sampling_when_temperature_positive():
    model = _make_model("답변")
    messages = [Message(role="user", content="질문")]
    model.generate(messages, temperature=0.7)
    call_kwargs = model._model.generate.call_args
    assert call_kwargs.kwargs.get("do_sample", call_kwargs[1].get("do_sample")) is True

import pytest

from legal_rag.llm import FakeChatModel, ModelResponse, StructuredOutputError
from legal_rag.rag.generator import GenerationError, Generator
from legal_rag.schemas import EvidenceBundle
from tests.legal_rag.unit.test_verification import _answer, _bundle


def test_generator_builds_messages_with_evidence_and_parses():
    model = FakeChatModel({"LegalAnswer": [_answer()]})
    out = Generator(model).generate("보증금을 못 받았어요", _bundle())
    assert out.summary.startswith("보증금")
    msgs = model.calls[0].messages
    assert msgs[0].role == "system" and "[LEGAL SOURCE S1]" in msgs[1].content
    assert msgs[-1].content == "질문: 보증금을 못 받았어요"


def test_generator_retries_once_on_structured_output_error():
    class Flaky:
        name = "flaky"

        def __init__(self):
            self.n = 0

        def generate(self, messages, **kwargs):
            self.n += 1
            if self.n == 1:
                raise StructuredOutputError("bad", "garbage")
            assert "스키마" in messages[-1].content
            return ModelResponse(text="{}", parsed=_answer())

    g = Generator(Flaky())
    assert g.generate("q", _bundle()).confidence
    assert g.model.n == 2


def test_generator_raises_after_retry_fails():
    class Dead:
        name = "dead"

        def generate(self, *args, **kwargs):
            raise StructuredOutputError("bad", "x")

    with pytest.raises(GenerationError):
        Generator(Dead()).generate("q", _bundle())


def test_feedback_is_appended_and_empty_bundle_is_stated():
    model = FakeChatModel({"LegalAnswer": [_answer()]})
    Generator(model).generate("q", EvidenceBundle(items=[]), feedback="근거 없는 인용")
    msgs = model.calls[0].messages
    assert msgs[-1].content == "근거 없는 인용" and "검색된 법률 근거가 없습니다" in msgs[1].content


def test_truncated_json_is_retried_with_a_request_for_a_shorter_answer():
    """A schema-repair prompt did not stop qwen3:4b from looping past max_tokens again (2026-09-28)."""
    class Truncated:
        name = "truncated"

        def __init__(self):
            self.n = 0

        def generate(self, messages, **kwargs):
            self.n += 1
            if self.n == 1:
                raise StructuredOutputError("LegalAnswer: Invalid JSON: EOF while parsing a string at line 5", '{"summary": "…')
            assert "잘렸습니다" in messages[-1].content and "짧게" in messages[-1].content
            return ModelResponse(text="{}", parsed=_answer())

    assert Generator(Truncated()).generate("q", _bundle()).confidence

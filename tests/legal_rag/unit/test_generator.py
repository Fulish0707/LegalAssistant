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


def test_figures_of_another_case_can_be_hidden_from_the_model():
    """A retrieved judgment's amounts and dates are another person's: the model put a judgment's
    "110,000,000원 ... 2023. 3. 12.부터" into an answer about the user's lease (2026-10-06)."""
    from legal_rag.schemas import DocumentType, Evidence

    bundle = EvidenceBundle(items=[
        Evidence(source_id="S1", chunk_id="c1", document_id="d1", document_type=DocumentType.judgment,
                 header={"Type": "Case", "Court": "서울중앙지방법원", "Case number": "2022가단12345", "Decision date": "2023-05-01"},
                 text="피고는 원고에게 110,000,000원 및 1억 4,000만 원에 대하여 2023. 3. 12.부터 2023년 4월 4일까지 연 5%의 비율로 계산한 돈을 지급하라. 2년의 기간.", score=0.9),
        Evidence(source_id="S2", chunk_id="c2", document_id="d2", document_type=DocumentType.statute,
                 header={"Type": "Statute", "Law": "주택임대차보호법", "Article": "제8조"},
                 text="보증금 중 5,000만 원 이하의 일정액. <개정 2013. 8. 13.>", score=0.8),
    ])
    shown = bundle.render(mask_case_figures=True)
    case, statute = shown.split("[LEGAL SOURCE S2]")
    assert "110,000,000" not in case and "4,000만" not in case and "2023. 3. 12." not in case and "4월 4일" not in case
    assert "(금액)" in case and "(날짜)" in case
    assert "2022가단12345" in case and "2023-05-01" in case and "연 5%" in case and "2년" in case
    assert "5,000만 원" in statute and "2013. 8. 13." in statute
    assert bundle.render() == bundle.render(mask_case_figures=False) and "110,000,000원" in bundle.render()

    model = FakeChatModel({"LegalAnswer": [_answer(), _answer()]})
    Generator(model, mask_case_figures=True).generate("q", bundle)
    Generator(model).generate("q", bundle)
    assert "110,000,000" not in model.calls[0].messages[1].content and "110,000,000" in model.calls[1].messages[1].content

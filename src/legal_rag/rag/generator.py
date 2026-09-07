"""Ask the model for a structured ``LegalAnswer`` over the evidence, with one repair retry."""

from __future__ import annotations

from legal_rag.llm import ChatModel, Message, StructuredOutputError
from legal_rag.llm.prompts import load_prompt
from legal_rag.schemas import EvidenceBundle, LegalAnswer


class GenerationError(RuntimeError):
    """The model could not produce a schema-valid answer even after a repair attempt."""


class Generator:
    def __init__(self, model: ChatModel, *, temperature: float = 0.2, max_tokens: int = 1024) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, message: str, bundle: EvidenceBundle, *, feedback: str | None = None) -> LegalAnswer:
        evidence = bundle.render() if bundle.items else "(검색된 법률 근거가 없습니다. 근거 없이 조문이나 판례를 언급하지 마세요.)"
        messages = [
            Message(role="system", content=load_prompt("answer_system")),
            Message(role="user", content="다음은 검색된 법률 근거입니다.\n\n" + evidence),
            Message(role="user", content="질문: " + message),
        ]
        if feedback:
            messages.append(Message(role="user", content=feedback))
        last_error: StructuredOutputError | None = None
        for _ in range(2):
            try:
                response = self.model.generate(messages, response_schema=LegalAnswer, temperature=self.temperature, max_tokens=self.max_tokens)
            except StructuredOutputError as exc:
                last_error = exc
                messages = messages + [Message(role="user", content=f"출력이 JSON 스키마에 맞지 않았습니다: {exc}. 스키마에 맞는 JSON 객체만 출력하세요.")]
                continue
            if isinstance(response.parsed, LegalAnswer):
                return response.parsed
            last_error = StructuredOutputError("model returned no parsed answer", response.text)
        raise GenerationError(str(last_error))

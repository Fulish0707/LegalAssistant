"""Ask the model for a structured ``LegalAnswer`` over the evidence, with one repair retry.

When *plain_text* mode is enabled the model is not asked for JSON at all; its
free-form answer is wrapped in a minimal ``LegalAnswer`` so the downstream
pipeline (verification, chat response) stays unchanged.  This is the right mode
for a QLoRA adapter trained on plain-text legal QA.
"""

from __future__ import annotations

from legal_rag.llm import ChatModel, Message, StructuredOutputError
from legal_rag.llm.prompts import load_prompt
from collections.abc import Sequence

from legal_rag.schemas import EvidenceBundle, LegalAnswer, Turn, UserDocument, render_user_documents

from .conversation import render_turns

_PLAIN_SYSTEM = (
    "당신은 한국 법률 정보를 안내하는 보조 상담 시스템입니다. "
    "변호사가 아니며, 확정적인 법률 자문이 아니라 근거에 기반한 법률 정보와 예비 안내를 제공합니다.\n\n"
    "제공되는 법률 자료를 참고하여 질문에 답변하세요. "
    "근거에 없는 법령 조문이나 판례를 만들어 내지 마세요. "
    "한국어로, 일반인이 이해할 수 있게 쓰되 법률 용어는 정확히 사용하세요."
)


def _repair_request(error: StructuredOutputError) -> str:
    # qwen3:4b sometimes loops inside "analysis" until max_tokens cuts the JSON off
    # (2 of 80 turns, 2026-09-28). Asking it to fix the schema just repeats the loop.
    if "EOF while parsing" in str(error):
        return ("출력이 너무 길어 JSON이 중간에 잘렸습니다. 같은 내용을 훨씬 짧게 쓰세요. "
                "analysis는 다섯 문장 이내로, 각 목록은 세 항목 이내로 쓰고 같은 문장을 되풀이하지 마세요. "
                "스키마에 맞는 JSON 객체만 출력하세요.")
    return f"출력이 JSON 스키마에 맞지 않았습니다: {error}. 스키마에 맞는 JSON 객체만 출력하세요."


class GenerationError(RuntimeError):
    """The model could not produce a schema-valid answer even after a repair attempt."""


class Generator:
    def __init__(self, model: ChatModel, *, temperature: float = 0.2, max_tokens: int = 1024, plain_text: bool = False,
                 mask_case_figures: bool = False, answer_style: str = "full") -> None:
        if answer_style not in ("full", "concise"):
            raise ValueError(f"unknown answer style {answer_style!r}")
        self.answer_style = answer_style
        self.mask_case_figures = mask_case_figures
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.plain_text = plain_text

    def generate(self, message: str, bundle: EvidenceBundle, *, feedback: str | None = None, history: list[Turn] | None = None,
                 open_questions: list[str] | None = None, user_documents: Sequence[UserDocument] = ()) -> LegalAnswer:
        if self.plain_text:
            return self._generate_plain(message, bundle, feedback=feedback)
        return self._generate_structured(message, bundle, feedback=feedback, history=history, open_questions=open_questions,
                                         user_documents=user_documents)

    # ── plain text mode (for QLoRA adapter) ─────────────────────────────────

    def _generate_plain(self, message: str, bundle: EvidenceBundle, *, feedback: str | None = None) -> LegalAnswer:
        evidence = self._render_evidence_plain(bundle)
        messages = [
            Message(role="system", content=_PLAIN_SYSTEM),
            Message(role="user", content=evidence + "\n\n질문: " + message),
        ]
        if feedback:
            messages.append(Message(role="user", content=feedback))
        response = self.model.generate(messages, temperature=self.temperature, max_tokens=self.max_tokens)
        return LegalAnswer(
            summary=response.text,
            legal_issues=[],
            analysis=response.text,
            possible_actions=[],
            citations=[],
            confidence="medium",
        )

    @staticmethod
    def _render_evidence_plain(bundle: EvidenceBundle) -> str:
        """Render evidence in the adapter's training format: ``[법률 자료]`` blocks."""
        if not bundle.items:
            return ""
        blocks = []
        for item in bundle.items:
            header_lines = [f"{k}: {v}" for k, v in item.header.items()]
            block = "[법률 자료]\n" + "\n".join(header_lines) + "\n\n" + item.text
            blocks.append(block)
        return "\n\n".join(blocks)

    # ── structured JSON mode (for Ollama / large models) ────────────────────

    def _generate_structured(self, message: str, bundle: EvidenceBundle, *, feedback: str | None = None, history: list[Turn] | None = None,
                             open_questions: list[str] | None = None, user_documents: Sequence[UserDocument] = ()) -> LegalAnswer:
        evidence = bundle.render(mask_case_figures=self.mask_case_figures) if bundle.items else "(검색된 법률 근거가 없습니다. 근거 없이 조문이나 판례를 언급하지 마세요.)"
        system = load_prompt("answer_system")
        if self.answer_style == "concise":
            system += "\n\n" + load_prompt("answer_concise")
        messages = [
            Message(role="system", content=system),
            Message(role="user", content="다음은 검색된 법률 근거입니다.\n\n" + evidence),
        ]
        if user_documents:
            # The rules travel with the documents, so a turn without any is prompted exactly as before.
            messages.append(Message(role="user", content=load_prompt("user_documents") + "\n\n" + render_user_documents(user_documents)))
        if history:
            # Context for what "그것" refers to. It is not evidence: citations are still checked against the sources.
            messages.append(Message(role="user", content="이전 대화(맥락 참고용이며 법적 근거가 아닙니다):\n" + "\n".join(render_turns(history))))
        if open_questions:
            messages.append(Message(role="user", content=(
                "아직 확인되지 않은 사실:\n" + "\n".join(f"- {item}" for item in open_questions) + "\n"
                "답변을 미루지 말고, 이 사실을 모르는 상태에서 근거로 답할 수 있는 범위까지 답하세요. "
                "어떤 경우를 전제로 한 설명인지 밝히고, 사실에 따라 결론이 달라지는 부분은 경우를 나누어 설명하세요. "
                "위 사실은 additional_information_needed에 사용자에게 물을 질문으로 담으세요.")))
        messages.append(Message(role="user", content="질문: " + message))
        if feedback:
            messages.append(Message(role="user", content=feedback))
        last_error: StructuredOutputError | None = None
        for _ in range(2):
            try:
                response = self.model.generate(messages, response_schema=LegalAnswer, temperature=self.temperature, max_tokens=self.max_tokens)
            except StructuredOutputError as exc:
                last_error = exc
                messages = messages + [Message(role="user", content=_repair_request(exc))]
                continue
            if isinstance(response.parsed, LegalAnswer):
                return response.parsed
            last_error = StructuredOutputError("model returned no parsed answer", response.text)
        raise GenerationError(str(last_error))

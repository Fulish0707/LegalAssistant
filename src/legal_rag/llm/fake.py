"""Scripted model for tests. Responses are keyed by the requested schema name (or "text")."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from .base import Message, ModelResponse, StructuredOutputError


@dataclass
class FakeCall:
    messages: list[Message]
    response_schema: type[BaseModel] | None
    temperature: float
    max_tokens: int = 1024


@dataclass
class FakeChatModel:
    responses: dict[str, list]
    name: str = "fake"
    calls: list[FakeCall] = field(default_factory=list)

    def generate(self, messages, *, response_schema=None, temperature=0.2, max_tokens=1024, on_text=None) -> ModelResponse:
        self.calls.append(FakeCall(list(messages), response_schema, temperature, max_tokens))
        key = response_schema.__name__ if response_schema else "text"
        queue = self.responses.get(key) or []
        if not queue:
            raise AssertionError(f"FakeChatModel has no scripted response for {key!r}")
        scripted = queue.pop(0)
        if on_text is not None:
            on_text(scripted.model_dump_json() if isinstance(scripted, BaseModel) else str(scripted))
        if isinstance(scripted, BaseModel):
            return ModelResponse(text=scripted.model_dump_json(), parsed=scripted)
        if response_schema is not None:
            try:
                parsed = response_schema.model_validate_json(scripted)
            except ValidationError as exc:  # as the real backends report a malformed answer
                raise StructuredOutputError(f"{response_schema.__name__}: {exc}", scripted) from exc
            return ModelResponse(text=scripted, parsed=parsed)
        return ModelResponse(text=str(scripted))

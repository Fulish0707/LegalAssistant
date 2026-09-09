"""Chat model protocol. Backends return text and, when asked, a parsed Pydantic object."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class ModelResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    text: str
    parsed: BaseModel | None = None
    raw: dict = Field(default_factory=dict)


class StructuredOutputError(RuntimeError):
    """The model answered, but not with JSON matching the requested schema."""

    def __init__(self, message: str, text: str) -> None:
        super().__init__(message)
        self.text = text


class ChatModel(Protocol):
    name: str

    def generate(
        self,
        messages: list[Message],
        *,
        response_schema: type[BaseModel] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> ModelResponse: ...

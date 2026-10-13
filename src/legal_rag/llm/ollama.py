"""Chat completions through Ollama's ``/api/chat`` with JSON-schema constrained output."""

from __future__ import annotations

import json
import logging
import re
import time

import httpx
from pydantic import BaseModel, ValidationError

from .base import ModelResponse, StructuredOutputError

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)
logger = logging.getLogger(__name__)


class OllamaChatModel:
    def __init__(self, model: str, base_url: str, timeout: float = 120.0, transport: httpx.BaseTransport | None = None, think: bool = False, num_ctx: int | None = None) -> None:
        self.model = model
        self.num_ctx = num_ctx
        self.name = f"ollama:{model}"
        self.think = think
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def generate(self, messages, *, response_schema: type[BaseModel] | None = None, temperature: float = 0.2, max_tokens: int = 1024) -> ModelResponse:
        payload: dict = {
            "model": self.model,
            "messages": [message.model_dump() for message in messages],
            "stream": False,
            "think": self.think,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if response_schema is not None:
            payload["format"] = response_schema.model_json_schema()
        if self.num_ctx is not None:
            payload["options"]["num_ctx"] = self.num_ctx
        started = time.perf_counter()
        response = self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        body = response.json()
        # Counters only: legal documents, questions, and model output stay out of logs.
        logger.info(
            "ollama_generation model=%s schema=%s wall_ms=%.1f total_ms=%.1f "
            "load_ms=%.1f prompt_ms=%.1f generation_ms=%.1f "
            "prompt_tokens=%s output_tokens=%s stop=%s",
            self.model, response_schema.__name__ if response_schema else "text",
            (time.perf_counter() - started) * 1000,
            (body.get("total_duration") or 0) / 1_000_000,
            (body.get("load_duration") or 0) / 1_000_000,
            (body.get("prompt_eval_duration") or 0) / 1_000_000,
            (body.get("eval_duration") or 0) / 1_000_000,
            body.get("prompt_eval_count"), body.get("eval_count"), body.get("done_reason"),
        )
        text = (body.get("message") or {}).get("content", "")
        if response_schema is None:
            return ModelResponse(text=text, raw=body)
        cleaned = _FENCE.sub(r"\1", text)
        try:
            parsed = response_schema.model_validate_json(cleaned)
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            raise StructuredOutputError(f"{response_schema.__name__}: {exc}", text) from exc
        return ModelResponse(text=text, parsed=parsed, raw=body)

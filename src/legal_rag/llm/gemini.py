"""Gemini GenerateContent adapter for the backend's chat model protocol."""

from __future__ import annotations

import re

import httpx
from pydantic import ValidationError

from .base import ModelResponse, StructuredOutputError


class GeminiChatModel:
    def __init__(self, model: str, api_key: str, timeout: float = 120.0,
                 transport: httpx.BaseTransport | None = None,
                 thinking_level: str | None = None) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("Gemini requires GEMINI_API_KEY in .env or the environment.")
        self.model = model.removeprefix("models/")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", self.model):
            raise ValueError("Invalid Gemini model ID.")
        self.name = f"gemini:{self.model}"
        self.thinking_level = thinking_level
        self._client = httpx.Client(
            base_url="https://generativelanguage.googleapis.com/v1beta/",
            headers={"x-goog-api-key": api_key}, timeout=timeout, transport=transport,
        )

    def generate(self, messages, *, response_schema=None, temperature=0.2, max_tokens=1024) -> ModelResponse:
        system = [message.content for message in messages if message.role == "system"]
        contents = []
        for message in messages:
            if message.role == "system":
                continue
            role = "model" if message.role == "assistant" else "user"
            part = {"text": message.content}
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].append(part)
            else:
                contents.append({"role": role, "parts": [part]})
        config = {"temperature": temperature, "maxOutputTokens": max_tokens}
        if self.thinking_level is not None:
            config["thinkingConfig"] = {"thinkingLevel": self.thinking_level}
        if response_schema is not None:
            config.update(responseMimeType="application/json",
                          responseJsonSchema=response_schema.model_json_schema())
        payload = {"contents": contents, "generationConfig": config}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": text} for text in system]}
        response = self._client.post(f"models/{self.model}:generateContent", json=payload)
        response.raise_for_status()
        body = response.json()
        candidates = body.get("candidates") or []
        parts = (candidates[0].get("content") or {}).get("parts", []) if candidates else []
        text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
        if not text.strip():
            reason = (body.get("promptFeedback") or {}).get("blockReason")
            reason = reason or (candidates[0].get("finishReason") if candidates else "no candidate")
            raise StructuredOutputError(f"Gemini returned no answer ({reason}).", text)
        if response_schema is None:
            return ModelResponse(text=text, raw=body)
        try:
            parsed = response_schema.model_validate_json(text)
        except (ValidationError, ValueError) as exc:
            raise StructuredOutputError(f"{response_schema.__name__}: {exc}", text) from exc
        return ModelResponse(text=text, parsed=parsed, raw=body)

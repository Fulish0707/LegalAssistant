"""Chat model backed by a local Transformers model with optional PEFT adapter.

Heavy imports (torch, transformers, peft, bitsandbytes) are all inside the class
so that importing this module alone costs nothing.  The model is loaded lazily on
the first ``generate`` call.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ValidationError

from .base import Message, ModelResponse, StructuredOutputError

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class TransformersChatModel:
    """4-bit Qwen model loaded via bitsandbytes, with optional LoRA adapter.

    Parameters
    ----------
    model_id:
        HuggingFace model id, e.g. ``"Qwen/Qwen3-4B"``.
    adapter:
        Path to a PEFT adapter directory.  ``None`` runs the base model.
    device_map:
        Passed to ``from_pretrained``.  Defaults to ``"auto"``.
    """

    def __init__(
        self,
        model_id: str,
        *,
        adapter: str | None = None,
        device_map: str = "auto",
    ) -> None:
        self._model_id = model_id
        self._adapter_path = adapter
        self._device_map = device_map
        self.name = f"transformers:{model_id}" + ("+adapter" if adapter else "")

        # Lazy-loaded on first generate()
        self._model = None
        self._tokenizer = None

    # ── lazy loading ────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        base = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            quantization_config=bnb,
            device_map=self._device_map,
        )

        if self._adapter_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(base, self._adapter_path)
        else:
            self._model = base

        self._model.eval()

    # ── generation ──────────────────────────────────────────────────────────

    def generate(
        self,
        messages: list[Message],
        *,
        response_schema: type[BaseModel] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        on_text=None,
    ) -> ModelResponse:
        import torch

        self._ensure_loaded()
        assert self._model is not None and self._tokenizer is not None

        chat_messages = [{"role": m.role, "content": m.content} for m in messages]
        prompt = self._tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs: dict = dict(
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9
        else:
            gen_kwargs["temperature"] = None
            gen_kwargs["top_p"] = None
            gen_kwargs["top_k"] = None

        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)

        text = self._tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)

        if response_schema is None:
            return ModelResponse(text=text)

        # Structured output: try to parse JSON from the text
        cleaned = _FENCE.sub(r"\1", text)
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            cleaned = cleaned[start : end + 1]
        try:
            parsed = response_schema.model_validate_json(cleaned)
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            raise StructuredOutputError(f"{response_schema.__name__}: {exc}", text) from exc
        return ModelResponse(text=text, parsed=parsed)

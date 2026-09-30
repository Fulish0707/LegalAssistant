"""Shared command-line plumbing so every script resolves settings the same way."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os

from dotenv import dotenv_values

from legal_rag.config import Settings, load_settings


def settings_from_options(
    *,
    profile: str | None = None,
    root: Path | None = None,
    raw_dir: Path | None = None,
    corpus_version: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Settings:
    # Read only chat settings from the shared Flask/RAG .env, without modifying
    # process environment. Explicit CLI options and exported settings take precedence.
    env_root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    dotenv = dotenv_values(env_root / ".env")
    llm = {key: dotenv[name] for key, name in (
        ("backend", "LEGAL_RAG__LLM__BACKEND"), ("model", "LEGAL_RAG__LLM__MODEL"),
        ("max_tokens", "LEGAL_RAG__LLM__MAX_TOKENS"),
        ("gemini_thinking_level", "LEGAL_RAG__LLM__GEMINI_THINKING_LEVEL"),
    ) if dotenv.get(name)}
    api_key = os.getenv("GEMINI_API_KEY") or dotenv.get("GEMINI_API_KEY")
    if api_key:
        llm["api_key"] = api_key
    overrides: dict[str, Any] = {"llm": llm} if llm else {}
    if root is not None:
        overrides.setdefault("paths", {})["root"] = str(Path(root).resolve())
    if raw_dir is not None:
        overrides.setdefault("paths", {})["raw_dir"] = str(Path(raw_dir).resolve())  # CLI paths are relative to the shell, not to root
    if corpus_version is not None:
        overrides["corpus"] = {"version": corpus_version}
    if extra:
        for section, values in extra.items():
            overrides.setdefault(section, {}).update(values)
    return load_settings(profile=profile, overrides=overrides)

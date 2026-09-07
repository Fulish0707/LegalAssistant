"""Shared command-line plumbing so every script resolves settings the same way."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from legal_rag.config import Settings, load_settings


def settings_from_options(
    *,
    profile: str | None = None,
    root: Path | None = None,
    raw_dir: Path | None = None,
    corpus_version: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Settings:
    overrides: dict[str, Any] = {}
    if root is not None:
        overrides.setdefault("paths", {})["root"] = str(root)
    if raw_dir is not None:
        overrides.setdefault("paths", {})["raw_dir"] = str(raw_dir)
    if corpus_version is not None:
        overrides["corpus"] = {"version": corpus_version}
    if extra:
        for section, values in extra.items():
            overrides.setdefault(section, {}).update(values)
    return load_settings(profile=profile, overrides=overrides)

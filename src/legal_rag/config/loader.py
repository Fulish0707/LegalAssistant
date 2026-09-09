"""Load layered YAML profiles into :class:`Settings`."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .settings import Settings

SECRET_PATHS: tuple[tuple[str, str], ...] = (("qdrant", "api_key"),)
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return data


def _reject_secrets(data: dict[str, Any], path: Path) -> None:
    for section, key in SECRET_PATHS:
        if isinstance(data.get(section), dict) and key in data[section]:
            raise ValueError(f"{path}: {section}.{key} must come from the environment, not YAML")


def load_settings(
    profile: str | None = None,
    config_dir: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Merge ``default.yaml``, ``<profile>.yaml``, explicit overrides, then let env win."""
    config_dir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    profile = profile or os.getenv("LEGAL_RAG_PROFILE", "local")
    merged: dict[str, Any] = {}
    for name in dict.fromkeys(("default", profile)):
        path = config_dir / f"{name}.yaml"
        data = _read_yaml(path)
        _reject_secrets(data, path)
        merged = _deep_merge(merged, data)
    if overrides:
        merged = _deep_merge(merged, overrides)
    return Settings(**merged)

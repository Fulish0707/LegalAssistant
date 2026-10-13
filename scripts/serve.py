#!/usr/bin/env python
"""Serve the HTTP API.

Example:
    uv run python scripts/serve.py --corpus-version v1 --port 8000
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
import uvicorn

from legal_rag.api import create_app
from legal_rag.cli import settings_from_options

app = typer.Typer(add_completion=False)


@app.command()
def main(
    profile: str = typer.Option(None),
    root: Path = typer.Option(None),
    corpus_version: str = typer.Option(None),
    host: str = typer.Option(None),
    port: int = typer.Option(None),
) -> None:
    settings = settings_from_options(profile=profile, root=root, corpus_version=corpus_version)
    # This namespace emits timing counters only; leave other libraries' log levels alone.
    timing_logger = logging.getLogger("legal_rag.llm.ollama")
    if not timing_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        timing_logger.addHandler(handler)
    timing_logger.setLevel(logging.INFO)
    timing_logger.propagate = False
    uvicorn.run(create_app(settings), host=host or settings.api.host, port=port or settings.api.port)


if __name__ == "__main__":
    app()

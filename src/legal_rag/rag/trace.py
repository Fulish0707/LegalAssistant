"""Record stage timings and decisions for one turn; optionally append a JSON line to a file."""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from legal_rag.schemas import StageTiming, Trace


class TraceRecorder:
    def __init__(self, *, conversation_id: str, user_text: str, include_text: bool = False, path: Path | None = None, model: str | None = None) -> None:
        self.path = Path(path) if path else None
        self.include_text = include_text
        self.trace = Trace(conversation_id=conversation_id, turn_id=uuid.uuid4().hex, model=model, user_text=user_text if include_text else None)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.trace.timings.append(StageTiming(stage=name, ms=round((time.perf_counter() - started) * 1000, 3)))

    def finish(self, *, policy_action: str) -> Trace:
        self.trace.policy_action = policy_action
        if not self.include_text:
            self.trace.analysis = None
            self.trace.search_queries = []
            self.trace.verification = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(self.trace.model_dump_json() + "\n")
        return self.trace

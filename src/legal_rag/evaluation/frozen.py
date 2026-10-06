"""Record a turn's analysis and retrieved chunks once, then replay them so only generation varies.

Two live runs of the same 80 questions shared identical evidence on 47 to 50 of them (2026-10-05),
so a single live run cannot separate a prompt or verifier change from retrieval noise. A frozen
turn runs the real ``Orchestrator.handle`` with its analyzer and retriever answering from the record.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from legal_rag.rag.generator import GenerationError
from legal_rag.schemas import QueryAnalysis, ScoredChunk


def capture_turn(orchestrator, question: str) -> dict:
    """Run one turn up to the evidence and return {question, analysis, hits, search_queries}.

    Generation is replaced by a failure the orchestrator already handles, so no answer is
    written and the model is called only by the analyzer.
    """
    retriever, generator, hits = orchestrator.retriever, orchestrator.generator, []
    originals = {name: getattr(retriever, name) for name in ("retrieve", "retrieve_many") if hasattr(retriever, name)}
    original_generate = generator.generate

    def spy(name):
        def call(*args, **kwargs):
            found = originals[name](*args, **kwargs)
            hits[:] = found
            return found
        return call

    def no_generation(*args, **kwargs):
        raise GenerationError("generation is skipped while capturing a frozen turn")

    for name in originals:
        setattr(retriever, name, spy(name))
    generator.generate = no_generation
    try:
        response = orchestrator.handle(question, conversation_id=uuid.uuid4().hex)
    finally:
        for name, original in originals.items():
            setattr(retriever, name, original)
        generator.generate = original_generate
    trace = response.trace
    return {
        "question": question,
        "analysis": trace.analysis.model_dump(mode="json") if trace and trace.analysis else {},
        "hits": [hit.model_dump(mode="json") for hit in hits],
        "search_queries": list(trace.search_queries) if trace else [],
        "stage_ms": {timing.stage: timing.ms for timing in trace.timings} if trace else {},
    }


def save_frozen(path: Path, rows: list[dict]) -> None:
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def load_frozen(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as handle:
        return {row["key"]: row for row in map(json.loads, handle)}


class FrozenRetriever:
    """Answers ``retrieve`` with the recorded hits of the loaded turn, and only for its question."""

    def __init__(self) -> None:
        self.turn: dict | None = None
        self.last_trace: dict = {}

    def load(self, turn: dict) -> None:
        self.turn = turn

    def retrieve(self, query: str, k: int, filters=None) -> list[ScoredChunk]:
        if self.turn is None or query != self.turn["question"]:
            raise ValueError(f"no frozen turn loaded for {query!r}")
        return [ScoredChunk.model_validate(hit) for hit in self.turn["hits"]][:k]


class FrozenAnalyzer:
    def __init__(self) -> None:
        self.turn: dict | None = None

    def load(self, turn: dict) -> None:
        self.turn = turn

    def analyze(self, message: str, history) -> QueryAnalysis:
        if self.turn is None or message != self.turn["question"]:
            raise ValueError(f"no frozen turn loaded for {message!r}")
        return QueryAnalysis.model_validate(self.turn["analysis"])

"""Score any ``Retriever`` on a retrieval dataset, independent of generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from legal_rag.retrieval.hybrid import Retriever
from legal_rag.schemas import RetrievalExample

from .retrieval_metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k


class RetrievalRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    query: str
    ranked_document_ids: list[str]
    recall: dict[int, float]
    precision: dict[int, float]
    mrr: float
    ndcg: dict[int, float]


class RetrievalReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retriever: str
    ks: list[int]
    rows: list[RetrievalRow] = Field(default_factory=list)
    aggregates: dict[str, float] = Field(default_factory=dict)

    def table(self) -> str:
        lines = [f"retriever={self.retriever} examples={len(self.rows)}"]
        lines += [f"{name:<14}{value:>8.4f}" for name, value in self.aggregates.items()]
        return "\n".join(lines)

    def write(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")


def ranked_document_ids(retriever: Retriever, example: RetrievalExample, depth: int) -> list[str]:
    hits = retriever.retrieve(example.query, depth, example.filters)
    seen: list[str] = []
    for hit in hits:
        if hit.chunk.document_id not in seen:
            seen.append(hit.chunk.document_id)
    return seen


def run_retrieval_eval(retriever: Retriever, examples: Sequence[RetrievalExample], ks: Sequence[int] = (5, 10, 20)) -> RetrievalReport:
    ks = sorted(set(ks))
    depth = max(ks) * 3
    rows: list[RetrievalRow] = []
    for example in examples:
        ranked = ranked_document_ids(retriever, example, depth)
        relevant = example.relevant_document_ids
        rows.append(
            RetrievalRow(
                id=example.id,
                query=example.query,
                ranked_document_ids=ranked[: max(ks)],
                recall={k: recall_at_k(ranked, relevant, k) for k in ks},
                precision={k: precision_at_k(ranked, relevant, k) for k in ks},
                mrr=mrr(ranked, relevant),
                ndcg={k: ndcg_at_k(ranked, relevant, k) for k in ks},
            )
        )
    n = len(rows) or 1
    aggregates: dict[str, float] = {}
    for k in ks:
        aggregates[f"recall@{k}"] = sum(row.recall[k] for row in rows) / n
    for k in ks:
        aggregates[f"precision@{k}"] = sum(row.precision[k] for row in rows) / n
    aggregates["mrr"] = sum(row.mrr for row in rows) / n
    for k in ks:
        aggregates[f"ndcg@{k}"] = sum(row.ndcg[k] for row in rows) / n
    return RetrievalReport(retriever=getattr(retriever, "name", type(retriever).__name__), ks=list(ks), rows=rows, aggregates=aggregates)

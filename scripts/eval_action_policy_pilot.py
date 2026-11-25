#!/usr/bin/env python
"""Fixed-evidence action-policy comparison; no vector indexes or private documents."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


class MeteredModel:
    """Record real generate calls including structured parses, timing and errors."""

    def __init__(self, model):
        self.model = model
        self.name = model.name
        self.calls = []

    def generate(self, messages, **kwargs):
        schema = kwargs.get("response_schema")
        row = {"schema": schema.__name__ if schema else None,
               "messages": [message.model_dump(mode="json") for message in messages],
               "temperature": kwargs.get("temperature"), "max_tokens": kwargs.get("max_tokens")}
        self.calls.append(row)
        started = time.perf_counter()
        try:
            response = self.model.generate(messages, **kwargs)
            row["output"] = response.model_dump(mode="json")
            return response
        except Exception as exc:
            row["error_type"] = type(exc).__name__
            if hasattr(exc, "text"):
                row["output_text"] = exc.text
            raise
        finally:
            row["elapsed_ms"] = (time.perf_counter() - started) * 1000


FICTIONAL_CASE_MARK = "실제 판례가 아님"


def load_sources(inputs):
    """Fixed evidence: official statute extracts, or windows of fictional judgments.

    A case source is a short window (``quote``) of a whole fictional judgment file. The
    window carries no heading, like an operational fixed-window chunk; the whole file is
    served through ``FixedRetriever.document_text`` as the vector stores would serve it.
    Real rulings are never stored here: every case file must say it is fictional.
    """
    from legal_rag.schemas import Chunk, ChunkMetadata, ScoredChunk

    inputs = Path(inputs)
    chunks, documents, hashes = {}, {}, {}
    for source in json.loads((inputs / "sources.json").read_text()):
        content = (inputs / source["extraction_file"]).read_text()
        digest = hashlib.sha256(content.encode()).hexdigest()
        if digest != source["sha256"] or source["quote"] not in content:
            raise ValueError("source extraction hash or literal quote mismatch")
        hashes[source["extraction_file"]] = digest
        if source.get("kind") == "case":
            if FICTIONAL_CASE_MARK not in content:
                raise ValueError("case source must be marked fictional: " + source["extraction_file"])
            chunk = Chunk(id=source["chunk_id"], document_id=source["id"], document_type="judgment",
                          path="window:0", order=0, text=source["quote"], title=source["title"],
                          metadata=ChunkMetadata(dataset="fictional-eval", corpus_version="action-policy-caselaw-v1",
                                                 court=source["court"], case_number=source["case_number"]))
            documents[source["chunk_id"]] = content
            chunks[source["id"]] = ScoredChunk(chunk=chunk, score=1, retriever="fixed-fictional")
            continue
        chunk = Chunk(id=source["chunk_id"], document_id=source["id"], document_type="statute",
                      path="art:" + source["article"], order=0, text=source["quote"],
                      title=source["law_name"] + " " + source["article"],
                      metadata=ChunkMetadata(dataset="official-law-go-kr", corpus_version="action-policy-pilot-v1",
                                             law_name=source["law_name"], article=source["article"],
                                             effective_date=source["effective_date"],
                                             original={"url": source["url"], "retrieved_at": source["retrieved_at"]}))
        chunks[source["id"]] = ScoredChunk(chunk=chunk, score=1, retriever="fixed-official")
    return chunks, documents, hashes


def run_comparison(settings, model, inputs, out, *, factory=None):
    from legal_rag.rag.orchestrator import Orchestrator
    from legal_rag.rag.conversation import InMemoryConversationStore
    from legal_rag.verification import Verifier, VerificationPolicy

    inputs, out = Path(inputs), Path(out)
    cases = json.loads((inputs / "cases.json").read_text())
    chunks, documents, hashes = load_sources(inputs)
    for name in ("cases.json", "sources.json", "retrieval.json"):
        if (inputs / name).exists():
            hashes[name] = hashlib.sha256((inputs / name).read_bytes()).hexdigest()
    out.mkdir(parents=True, exist_ok=False)
    # Freeze actual bytes with this run; later curation never changes a prior manifest.
    frozen = out / "inputs"
    frozen.mkdir()
    for name in hashes:
        (frozen / name).write_bytes((inputs / name).read_bytes())
    config = settings.model_dump(mode="json", exclude={"llm": {"api_key"}, "qdrant": {"api_key"}})
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "model_name": model.name,
                "settings": config, "input_hashes": hashes,
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                "protocol": "paired fixed first/extra bundles; fresh per-arm stores; alternating execution order",
                "human_legal_review": False}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    build = factory or Orchestrator.from_components
    rows = []
    with (out / "results.jsonl").open("x") as stream:
        for index, case in enumerate(cases):
            arms = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
            for arm in arms:
                retriever = FixedRetriever([chunks[key] for key in case["first"]], [chunks[key] for key in case["extra"]],
                                           documents=documents)
                metered = MeteredModel(model)
                orch = build(settings=settings, model=metered, retriever=retriever,
                             store=InMemoryConversationStore(), verifier=Verifier(),
                             policy=VerificationPolicy(max_regenerations=settings.rag.max_regenerations))
                row = run_arm(case, arm, orch, retriever, metered)
                row["manual_expected_actions"] = case["manual_expected_actions"]
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(json.dumps({key: row[key] for key in ("case_id", "arm", "status", "elapsed_ms")}), flush=True)
    report = {"attempted_arms": len(rows), "completed_arms": sum(row["status"] == "completed" for row in rows),
              "technical_errors": sum(row["status"] != "completed" for row in rows),
              "assessment_errors": sum(row["status"] == "assessment_error" for row in rows),
              "human_legal_review": False}
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    from legal_rag.config import load_settings
    from legal_rag.factory import build_chat_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "data/eval/action_policy_pilot")
    parser.add_argument("--out", type=Path, required=True, help="New directory, never overwritten")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--backend", choices=("ollama", "gemini", "transformers"))
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    args = parser.parse_args()
    settings = load_settings(profile=args.profile)
    llm_values = {key: value for key, value in {"backend": args.backend, "model": args.model,
                  "base_url": args.base_url}.items() if value is not None}
    # Explicit CLI overrides must beat environment, without serializing any secret.
    llm = settings.llm.model_copy(update=llm_values)
    if llm.backend == "fake":
        parser.error("fake backend is for tests, not live comparison")
    settings = settings.model_copy(update={"llm": llm, "trace": settings.trace.model_copy(update={"path": None})})
    print(json.dumps({"backend": llm.backend, "model": llm.model, "base_url": llm.base_url}), flush=True)
    report = run_comparison(settings, build_chat_model(settings), args.inputs, args.out)
    return 1 if report["technical_errors"] else 0


class FixedRetriever:
    """Replay first and extra evidence, retaining immutable original chunk IDs."""

    name = "fixed-official"

    def __init__(self, first, extra, *, documents=None):
        self.bundles = (first, extra)
        self.documents = documents or {}
        self.calls = []
        self.last_trace = {}

    def document_text(self, chunk_id):
        """Whole fictional document for a case window, as HybridRetriever.document_text serves it."""
        return self.documents.get(chunk_id)

    def retrieve(self, query, k, filters=None):
        if len(self.calls) >= 2:
            raise RuntimeError("fixed pilot permits only two retrieval calls")
        hits = self.bundles[len(self.calls)][:k]
        self.calls.append({"query": query, "k": k, "filters": filters,
                           "chunk_ids": [hit.chunk.id for hit in hits]})
        self.last_trace = {"fixed": [(hit.chunk.id, hit.score) for hit in hits]}
        return list(hits)


def run_arm(case, arm, orchestrator, retriever, model):
    import time
    import uuid
    from legal_rag.schemas import UserDocument

    kwargs = {"conversation_id": uuid.uuid4().hex,
              "user_documents": [UserDocument.model_validate(doc) for doc in case["user_documents"]]}
    if arm == "candidate" and case.get("policy_enabled", True):
        kwargs.update(policy_scope={"profile": "lease_review_v1", "matter_id": "fictional-lease-1",
                                    "issue_id": case["issue_id"]}, request_id=uuid.uuid4().hex)
    started = time.perf_counter()
    row = {"case_id": case["id"], "arm": arm, "response": None, "status": "completed"}
    try:
        response = orchestrator.handle(case["message"], **kwargs)
        row["response"] = response.model_dump(mode="json")
        if response.withheld and response.withheld.reason_code == "assessment_failed":
            row["status"] = "assessment_error"
    except Exception as exc:
        row.update(status="technical_error", error_type=type(exc).__name__)
    row.update(elapsed_ms=(time.perf_counter() - started) * 1000,
               retrieval_calls=retriever.calls, model_calls=model.calls)
    return row


if __name__ == "__main__":
    raise SystemExit(main())

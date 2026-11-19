"""Runner mechanics only: fake responses are never live semantic evidence."""
import importlib.util
from pathlib import Path

import pytest
from legal_rag.schemas import Chunk, ChunkMetadata, ScoredChunk

ROOT = Path(__file__).resolve().parents[3]


def pilot():
    path = ROOT / "scripts/eval_action_policy_pilot.py"
    assert path.exists(), "pilot runner has not been implemented"
    spec = importlib.util.spec_from_file_location("action_pilot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hit(identifier):
    return ScoredChunk(chunk=Chunk(id=identifier, document_id="law", document_type="statute",
        path="art:제105조", order=0, text="extracted law", title="민법",
        metadata=ChunkMetadata(dataset="official", corpus_version="pilot")), score=1, retriever="fixed")


def test_fixed_retrieval_preserves_ids_and_replays_only_two_bundles():
    m = pilot()
    retriever = m.FixedRetriever([hit("first")], [hit("extra")])
    assert retriever.retrieve("q", 5)[0].chunk.id == "first"
    assert retriever.retrieve("targeted", 5)[0].chunk.id == "extra"
    with pytest.raises(RuntimeError, match="two"):
        retriever.retrieve("third", 5)
    assert [row["chunk_ids"] for row in retriever.calls] == [["first"], ["extra"]]


def test_run_arm_preserves_legacy_kwargs_and_reports_technical_failure():
    from types import SimpleNamespace
    m = pilot()
    received = []
    class Orch:
        def handle(self, message, **kwargs):
            received.append(kwargs)
            raise RuntimeError("technical failure")
    case = {"id": "c", "message": "q", "issue_id": "rent-deduction", "user_documents": []}
    baseline = m.run_arm(case, "baseline", Orch(), SimpleNamespace(calls=[]), SimpleNamespace(calls=[]))
    candidate = m.run_arm(case, "candidate", Orch(), SimpleNamespace(calls=[]), SimpleNamespace(calls=[]))
    assert "policy_scope" not in received[0] and "request_id" not in received[0]
    assert received[1]["policy_scope"]["profile"] == "lease_review_v1"
    assert baseline["status"] == candidate["status"] == "technical_error"
    assert baseline["elapsed_ms"] >= 0
    assert baseline["response"] is None


def test_comparison_freezes_inputs_and_refuses_overwrite(tmp_path):
    from types import SimpleNamespace
    from legal_rag.config import load_settings
    from legal_rag.schemas import ChatResponse
    m = pilot()
    settings = load_settings()
    created = []
    def factory(**kwargs):
        created.append(kwargs)
        class Orch:
            def handle(self, message, **kw):
                kwargs["retriever"].retrieve(message, 5)
                return ChatResponse(kind="clarification", conversation_id=kw["conversation_id"], questions=["test only"])
        return Orch()
    out = tmp_path / "new-run"
    report = m.run_comparison(settings, SimpleNamespace(name="test-only"), ROOT / "data/eval/action_policy_pilot", out, factory=factory)
    assert report["attempted_arms"] == 10
    assert report["technical_errors"] == 0
    assert len(created) == 10
    import json
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 10
    assert rows[0]["retrieval_calls"][0]["chunk_ids"] == rows[1]["retrieval_calls"][0]["chunk_ids"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["input_hashes"]["sources.json"]
    assert "api_key" not in json.dumps(manifest)
    with pytest.raises(FileExistsError):
        m.run_comparison(settings, SimpleNamespace(name="test-only"), ROOT / "data/eval/action_policy_pilot", out, factory=factory)


def test_meter_records_raw_outputs_and_failures():
    from legal_rag.llm.base import ModelResponse, Message
    m = pilot()
    class Model:
        name = "test-only"
        def generate(self, messages, **kwargs):
            if messages[0].content == "fail":
                raise RuntimeError("test")
            return ModelResponse(text="raw output", raw={"test": True})
    model = m.MeteredModel(Model())
    model.generate([Message(role="user", content="ok")])
    with pytest.raises(RuntimeError):
        model.generate([Message(role="user", content="fail")])
    assert model.calls[0]["output"]["text"] == "raw output"
    assert model.calls[1]["error_type"] == "RuntimeError"
    assert all(row["elapsed_ms"] >= 0 for row in model.calls)


def test_default_nonlease_control_never_enables_policy_and_marks_assessment_failure():
    from types import SimpleNamespace
    from legal_rag.schemas import ChatResponse
    m = pilot()
    received = []
    class Orch:
        def handle(self, message, **kw):
            received.append(kw)
            return ChatResponse(kind="withheld", conversation_id=kw["conversation_id"], withheld={
                "reason_code": "assessment_failed", "explanation": "technical assessment failure"})
    case = {"id": "general-law-default", "message": "q", "issue_id": None,
            "policy_enabled": False, "user_documents": []}
    row = m.run_arm(case, "candidate", Orch(), SimpleNamespace(calls=[]), SimpleNamespace(calls=[]))
    assert "policy_scope" not in received[0]
    assert row["status"] == "assessment_error"


def test_retriever_trace_matches_schema():
    m = pilot()
    retriever = m.FixedRetriever([hit("first")], [])
    retriever.retrieve("q", 5)
    assert retriever.last_trace == {"fixed": [("first", 1.0)]}

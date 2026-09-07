import json
from datetime import datetime

from legal_rag.rag.conversation import InMemoryConversationStore, SQLiteConversationStore
from legal_rag.rag.trace import TraceRecorder
from legal_rag.schemas import Turn


def test_in_memory_store_appends_and_creates_on_get():
    s = InMemoryConversationStore()
    assert s.get("c1").turns == []
    s.append("c1", Turn(role="user", content="hi", created_at=datetime(2026, 1, 1)))
    assert [t.content for t in s.get("c1").turns] == ["hi"]
    assert s.get("other").turns == []


def test_sqlite_store_persists_in_order(tmp_path):
    path = tmp_path / "conv.db"
    store = SQLiteConversationStore(path)
    store.append("c1", Turn(role="user", content="q", created_at=datetime(2026, 1, 1)))
    store.append("c1", Turn(role="assistant", content="a", created_at=datetime(2026, 1, 1), evidence_ids=["x#y"]))
    turns = SQLiteConversationStore(path).get("c1").turns
    assert [t.role for t in turns] == ["user", "assistant"] and turns[1].evidence_ids == ["x#y"]


def test_trace_recorder_times_stages_and_writes_json_line(tmp_path):
    rec = TraceRecorder(conversation_id="c1", user_text="비밀", include_text=False, path=tmp_path / "trace.jsonl", model="fake")
    with rec.stage("analyze"):
        pass
    rec.trace.search_queries = ["q"]
    rec.finish(policy_action="accept")
    line = json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert line["timings"][0]["stage"] == "analyze" and line["user_text"] is None and line["policy_action"] == "accept"
    assert line["model"] == "fake" and line["search_queries"] == ["q"]
    assert TraceRecorder(conversation_id="c", user_text="보임", include_text=True).trace.user_text == "보임"

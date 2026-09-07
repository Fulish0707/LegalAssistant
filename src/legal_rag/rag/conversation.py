"""Conversation stores. History is kept apart from legal evidence by construction."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Protocol

from legal_rag.schemas import Conversation, Turn


class ConversationStore(Protocol):
    def get(self, conversation_id: str) -> Conversation: ...

    def append(self, conversation_id: str, turn: Turn) -> None: ...


class InMemoryConversationStore:
    def __init__(self) -> None:
        self._turns: dict[str, list[Turn]] = {}

    def get(self, conversation_id: str) -> Conversation:
        return Conversation(id=conversation_id, turns=list(self._turns.get(conversation_id, [])))

    def append(self, conversation_id: str, turn: Turn) -> None:
        self._turns.setdefault(conversation_id, []).append(turn)


class SQLiteConversationStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS turns (conversation_id TEXT NOT NULL, seq INTEGER NOT NULL, payload TEXT NOT NULL, PRIMARY KEY (conversation_id, seq))"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def get(self, conversation_id: str) -> Conversation:
        with self._connect() as conn:
            rows = conn.execute("SELECT payload FROM turns WHERE conversation_id=? ORDER BY seq", (conversation_id,)).fetchall()
        return Conversation(id=conversation_id, turns=[Turn.model_validate_json(row[0]) for row in rows])

    def append(self, conversation_id: str, turn: Turn) -> None:
        with self._connect() as conn:
            seq = conn.execute("SELECT COALESCE(MAX(seq), -1) + 1 FROM turns WHERE conversation_id=?", (conversation_id,)).fetchone()[0]
            conn.execute("INSERT INTO turns VALUES (?, ?, ?)", (conversation_id, seq, turn.model_dump_json()))

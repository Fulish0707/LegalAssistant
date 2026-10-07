"""Conversation stores. History is kept apart from legal evidence by construction."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Protocol

from legal_rag.schemas import Conversation, Turn


def render_turns(turns: list[Turn], *, chars: int = 500) -> list[str]:
    """One line per turn, each cut to `chars`: enough to resolve "그 신청", not enough to carry an answer."""
    lines = []
    for turn in turns:
        text = " ".join(turn.content.split())
        lines.append(f"- {'사용자' if turn.role == 'user' else '상담'}: {text[:chars]}{'…' if len(text) > chars else ''}")
    return lines


class ConversationStore(Protocol):
    def get(self, conversation_id: str) -> Conversation: ...

    def append(self, conversation_id: str, turn: Turn) -> None: ...

    def delete_prefix(self, prefix: str) -> None: ...

    def has_capacity(self, prefix: str, needed: int = 2) -> bool: ...


class InMemoryConversationStore:
    def __init__(self) -> None:
        self._turns: dict[str, list[Turn]] = {}

    def get(self, conversation_id: str) -> Conversation:
        return Conversation(id=conversation_id, turns=list(self._turns.get(conversation_id, [])[-100:]))

    def append(self, conversation_id: str, turn: Turn) -> None:
        self._turns.setdefault(conversation_id, []).append(turn)

    def delete_prefix(self, prefix: str) -> None:
        for key in list(self._turns):
            if key.startswith(prefix):
                del self._turns[key]

    def has_capacity(self, prefix: str, needed: int = 2) -> bool:
        return sum(len(turns) for key, turns in self._turns.items() if key.startswith(prefix)) + needed <= 5000


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
            rows = conn.execute("SELECT payload FROM turns WHERE conversation_id=? ORDER BY seq DESC LIMIT 100", (conversation_id,)).fetchall()[::-1]
        return Conversation(id=conversation_id, turns=[Turn.model_validate_json(row[0]) for row in rows])

    def append(self, conversation_id: str, turn: Turn) -> None:
        with self._connect() as conn:
            seq = conn.execute("SELECT COALESCE(MAX(seq), -1) + 1 FROM turns WHERE conversation_id=?", (conversation_id,)).fetchone()[0]
            conn.execute("INSERT INTO turns VALUES (?, ?, ?)", (conversation_id, seq, turn.model_dump_json()))

    def delete_prefix(self, prefix: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM turns WHERE conversation_id>=? AND conversation_id<?",
                         (prefix, prefix[:-1] + chr(ord(prefix[-1]) + 1)))

    def has_capacity(self, prefix: str, needed: int = 2) -> bool:
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM turns WHERE conversation_id>=? AND conversation_id<?",
                                 (prefix, prefix[:-1] + chr(ord(prefix[-1]) + 1))).fetchone()[0]
        return count + needed <= 5000

"""Atomic policy budgets, separate from conversation history and legal evidence."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from threading import Lock
from typing import Protocol


class PolicyRequestError(RuntimeError):
    """A request cannot safely proceed."""


class RequestConflict(PolicyRequestError):
    """The request identifier was reused with a different fingerprint."""


class RequestInProgress(PolicyRequestError):
    """The identical request is already running."""


class RequestFailed(PolicyRequestError):
    """The request failed; automatic replay must not retry its side effects."""


def _replay(fingerprint: str, row: tuple) -> str:
    if fingerprint != row[0]:
        raise RequestConflict("Request fingerprint does not match")
    if row[1] == "running":
        raise RequestInProgress("Request is already running")
    if row[1] == "failed":
        raise RequestFailed("Request previously failed")
    return row[2]


class PolicyStateStore(Protocol):
    """Reserve before side effects; reservations are never refunded on failure.

    Claim before reserving. Clarification's legacy three-key signature can only
    check that *some* request in the conversation is running; the caller must
    ensure its own request was claimed and is still running. Fingerprints and
    response JSON are opaque caller-owned strings; replay preserves exact text.
    """

    def claim_request(self, conversation_id: str, request_id: str, fingerprint: str) -> str | None: ...
    def reserve_clarification(self, conversation_id: str, matter_id: str, issue_id: str) -> bool: ...
    def clarification_used(self, conversation_id: str, matter_id: str, issue_id: str) -> bool: ...
    def reserve_research(self, conversation_id: str, request_id: str) -> bool: ...
    def research_used(self, conversation_id: str, request_id: str) -> bool: ...
    def complete_request(self, conversation_id: str, request_id: str, response_json: str) -> None: ...
    def fail_request(self, conversation_id: str, request_id: str) -> None: ...


class InMemoryPolicyStateStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._requests: dict[tuple[str, str], tuple[str, str, str | None]] = {}
        self._research: set[tuple[str, str]] = set()
        self._clarifications: set[tuple[str, str, str]] = set()

    def reserve_clarification(self, conversation_id: str, matter_id: str, issue_id: str) -> bool:
        with self._lock:
            if not any(key[0] == conversation_id and row[1] == "running"
                       for key, row in self._requests.items()):
                raise PolicyRequestError("Conversation must have a claimed running request")
            key = (conversation_id, matter_id, issue_id)
            if key in self._clarifications:
                return False
            self._clarifications.add(key)
            return True

    def clarification_used(self, conversation_id: str, matter_id: str, issue_id: str) -> bool:
        with self._lock:
            return (conversation_id, matter_id, issue_id) in self._clarifications

    def reserve_research(self, conversation_id: str, request_id: str) -> bool:
        with self._lock:
            key = (conversation_id, request_id)
            row = self._requests.get(key)
            if row is None or row[1] != "running":
                raise PolicyRequestError("Request must be claimed and running")
            if key in self._research:
                return False
            self._research.add(key)
            return True

    def research_used(self, conversation_id: str, request_id: str) -> bool:
        with self._lock:
            return (conversation_id, request_id) in self._research

    def claim_request(self, conversation_id: str, request_id: str, fingerprint: str) -> str | None:
        with self._lock:
            key = (conversation_id, request_id)
            if key in self._requests:
                return _replay(fingerprint, self._requests[key])
            self._requests[key] = (fingerprint, "running", None)
            return None

    def complete_request(self, conversation_id: str, request_id: str, response_json: str) -> None:
        self._transition(conversation_id, request_id, "completed", response_json)

    def fail_request(self, conversation_id: str, request_id: str) -> None:
        self._transition(conversation_id, request_id, "failed", None)

    def _transition(self, conversation_id: str, request_id: str, status: str, response: str | None) -> None:
        with self._lock:
            key = (conversation_id, request_id)
            row = self._requests.get(key)
            if row is not None and row[1] == status == "failed":
                return
            if row is None or row[1] != "running":
                raise PolicyRequestError("Request must be claimed and running")
            self._requests[key] = (row[0], status, response)


class SQLitePolicyStateStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS policy_requests (
                conversation_id TEXT NOT NULL, request_id TEXT NOT NULL,
                fingerprint TEXT NOT NULL, status TEXT NOT NULL,
                response_json TEXT,
                PRIMARY KEY (conversation_id, request_id))""")
            conn.execute("""CREATE TABLE IF NOT EXISTS policy_research (
                conversation_id TEXT NOT NULL, request_id TEXT NOT NULL,
                PRIMARY KEY (conversation_id, request_id))""")
            conn.execute("""CREATE TABLE IF NOT EXISTS policy_clarifications (
                conversation_id TEXT NOT NULL, matter_id TEXT NOT NULL,
                issue_id TEXT NOT NULL,
                PRIMARY KEY (conversation_id, matter_id, issue_id))""")

    def reserve_clarification(self, conversation_id: str, matter_id: str, issue_id: str) -> bool:
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT 1 FROM policy_requests WHERE conversation_id=? AND status='running' LIMIT 1",
                (conversation_id,),
            ).fetchone()
            if row is None:
                raise PolicyRequestError("Conversation must have a claimed running request")
            return conn.execute(
                "INSERT OR IGNORE INTO policy_clarifications VALUES (?, ?, ?)",
                (conversation_id, matter_id, issue_id),
            ).rowcount == 1

    def clarification_used(self, conversation_id: str, matter_id: str, issue_id: str) -> bool:
        with closing(self._connect()) as conn:
            return conn.execute(
                "SELECT 1 FROM policy_clarifications WHERE conversation_id=? AND matter_id=? AND issue_id=?",
                (conversation_id, matter_id, issue_id),
            ).fetchone() is not None

    def reserve_research(self, conversation_id: str, request_id: str) -> bool:
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status FROM policy_requests WHERE conversation_id=? AND request_id=?",
                (conversation_id, request_id),
            ).fetchone()
            if row is None or row[0] != "running":
                raise PolicyRequestError("Request must be claimed and running")
            return conn.execute(
                "INSERT OR IGNORE INTO policy_research VALUES (?, ?)",
                (conversation_id, request_id),
            ).rowcount == 1

    def research_used(self, conversation_id: str, request_id: str) -> bool:
        with closing(self._connect()) as conn:
            return conn.execute(
                "SELECT 1 FROM policy_research WHERE conversation_id=? AND request_id=?",
                (conversation_id, request_id),
            ).fetchone() is not None

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def complete_request(self, conversation_id: str, request_id: str, response_json: str) -> None:
        self._transition(conversation_id, request_id, "completed", response_json)

    def fail_request(self, conversation_id: str, request_id: str) -> None:
        self._transition(conversation_id, request_id, "failed", None)

    def _transition(self, conversation_id: str, request_id: str, status: str, response: str | None) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status FROM policy_requests WHERE conversation_id=? AND request_id=?",
                (conversation_id, request_id),
            ).fetchone()
            if row is not None and row[0] == status == "failed":
                return
            if row is None or row[0] != "running":
                raise PolicyRequestError("Request must be claimed and running")
            conn.execute(
                "UPDATE policy_requests SET status=?, response_json=? WHERE conversation_id=? AND request_id=?",
                (status, response, conversation_id, request_id),
            )

    def claim_request(self, conversation_id: str, request_id: str, fingerprint: str) -> str | None:
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT fingerprint, status, response_json FROM policy_requests WHERE conversation_id=? AND request_id=?",
                (conversation_id, request_id),
            ).fetchone()
            if row is not None:
                return _replay(fingerprint, row)
            conn.execute(
                "INSERT INTO policy_requests VALUES (?, ?, ?, 'running', NULL)",
                (conversation_id, request_id, fingerprint),
            )
            return None

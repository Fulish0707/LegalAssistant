"""SQLite idempotency for the explicitly enabled web pilot, including quota/history.

The backend independently protects model/search effects. These records protect
web POST/SSE effects. Interrupted running requests are not silently rerun.
"""
import hashlib
import json


def fingerprint(message, issue_id):
    return hashlib.sha256(json.dumps([message, issue_id], ensure_ascii=False).encode()).hexdigest()


def _ensure(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS pilot_web_requests (
        owner TEXT NOT NULL, scope TEXT NOT NULL, request_id TEXT NOT NULL,
        fingerprint TEXT NOT NULL, issue_id TEXT NOT NULL, status TEXT NOT NULL,
        final_json TEXT, PRIMARY KEY(owner, scope, request_id))""")


def submit(conn, owner, scope, request_id, message, issue_id, conversation):
    _ensure(conn)
    digest = fingerprint(message, issue_id)
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT fingerprint FROM pilot_web_requests WHERE owner=? AND scope=? AND request_id=?",
                           (owner, scope, request_id)).fetchone()
        if row is not None:
            if row[0] != digest:
                raise ValueError("pilot_request_conflict")
            return
        conn.execute("INSERT INTO pilot_web_requests VALUES (?, ?, ?, ?, ?, 'pending', NULL)",
                     (owner, scope, request_id, digest, issue_id))
        conn.execute("INSERT INTO chat_history (user_id, role, content, conversation) VALUES (?, 'user', ?, ?)",
                     (owner, message, conversation))


def lookup(conn, owner, scope, request_id, message):
    _ensure(conn)
    row = conn.execute("SELECT fingerprint, issue_id FROM pilot_web_requests WHERE owner=? AND scope=? AND request_id=?",
                       (owner, scope, request_id)).fetchone()
    if row is None or row[0] != fingerprint(message, row[1]):
        raise ValueError("pilot_request_conflict")
    return {"request_id": request_id, "issue_id": row[1], "key": scope}


def claim(conn, owner, scope, request_id):
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT status, final_json FROM pilot_web_requests WHERE owner=? AND scope=? AND request_id=?",
                           (owner, scope, request_id)).fetchone()
        if row is None:
            raise ValueError("pilot_request_missing")
        if row[0] == "pending":
            conn.execute("UPDATE pilot_web_requests SET status='running' WHERE owner=? AND scope=? AND request_id=?",
                         (owner, scope, request_id))
            return None
        if row[1] is not None:
            return json.loads(row[1])
        return {"type": "error", "error": "policy_request_in_progress",
                "content": "이 요청은 아직 처리 중이거나 중단된 상태입니다. 같은 요청을 다시 실행하지 않습니다."}


def finish(conn, owner, scope, request_id, conversation, event):
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT status FROM pilot_web_requests WHERE owner=? AND scope=? AND request_id=?",
                           (owner, scope, request_id)).fetchone()
        if row is None or row[0] != "running":
            raise ValueError("pilot_request_not_running")
        text = event.get("chat_response") if event["type"] == "final_response" else event.get("content")
        if text:
            conn.execute("INSERT INTO chat_history (user_id, role, content, conversation, sources) VALUES (?, 'assistant', ?, ?, ?)",
                         (owner, text, conversation, json.dumps(event.get("sources", []), ensure_ascii=False)))
        conn.execute("UPDATE pilot_web_requests SET status=?, final_json=? WHERE owner=? AND scope=? AND request_id=?",
                     ("completed" if event["type"] == "final_response" else "failed",
                      json.dumps(event, ensure_ascii=False), owner, scope, request_id))

# Anonymous User Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Isolate drafts, chats, model context, and uploaded RAG data by anonymous browser session.

**Architecture:** Flask creates a signed-session `user_id`. SQLite queries and Chroma metadata use that explicit ID; chat context comes from that user's database rows rather than global memory.

**Tech Stack:** Flask sessions, SQLite, Chroma/LangChain, Python `unittest`.

---

### Task 1: Assign an anonymous session identity

**Files:**

- Modify: `app.py:1-96`
- Test: `tests/test_app_contract.py`

- [ ] **Step 1: Write this failing two-client test.**

```python
def test_clients_receive_distinct_anonymous_identities(self):
    first = self.app_module.app.test_client()
    second = self.app_module.app.test_client()
    first.get("/health")
    second.get("/health")
    with first.session_transaction() as data:
        first_id = data["user_id"]
    with second.session_transaction() as data:
        second_id = data["user_id"]
    self.assertNotEqual(first_id, second_id)
```

- [ ] **Step 2: Run `python -m unittest tests.test_app_contract.AppContractTests.test_clients_receive_distinct_anonymous_identities -v`; expect a missing-key failure.**
- [ ] **Step 3: Add `secrets`, `session`, `app.config["SECRET_KEY"] = os.getenv("SECRET_KEY") or secrets.token_urlsafe(32)`, and a `before_request` hook that creates `session["user_id"]` with `secrets.token_urlsafe(32)` only when absent.**
- [ ] **Step 4: Re-run the focused test; expect PASS.**
- [ ] **Step 5: Commit with `git add app.py tests/test_app_contract.py && git commit -m "feat: assign anonymous session identities"`.**

### Task 2: Migrate and isolate SQLite state

**Files:**

- Modify: `app.py:44-175, 495-556`
- Test: `tests/test_app_contract.py`

- [ ] **Step 1: Write this failing draft-isolation test.**

```python
def test_drafts_are_session_scoped(self):
    first = self.app_module.app.test_client()
    second = self.app_module.app.test_client()
    first.post("/update_statement", json={"personal_statement_draft": "First"})
    second.post("/update_statement", json={"personal_statement_draft": "Second"})
    self.assertEqual(first.get("/get_initial_state").json["personal_statement_draft"], "First")
    self.assertEqual(second.get("/get_initial_state").json["personal_statement_draft"], "Second")
```

- [ ] **Step 2: Run it; expect failure because the singleton draft is shared.**
- [ ] **Step 3: Transactionally migrate legacy tables into schemas with `personal_statement(user_id TEXT PRIMARY KEY, draft TEXT NOT NULL DEFAULT '')` and `chat_history(..., user_id TEXT NOT NULL, ...)`; assign old content to `legacy-unassigned`.**
- [ ] **Step 4: Require `user_id` for every draft/chat list, insert, search, update, and clear helper; pass `session["user_id"]` from each route.**
- [ ] **Step 5: Add chat-list, search, clear, and legacy-migration tests; run `python -m unittest discover -s tests -v`; expect PASS.**
- [ ] **Step 6: Commit with `git add app.py tests/test_app_contract.py && git commit -m "feat: isolate SQLite state by anonymous user"`.**

### Task 3: Isolate RAG and chat context

**Files:**

- Modify: `app.py:325-556, 583-731`
- Modify: `langchain_config.py:94-208`
- Test: `tests/test_app_contract.py`

- [ ] **Step 1: Write a failing upload test asserting `captured["metadata"]["user_id"]` exists and differs across two clients.**
- [ ] **Step 2: Run it; expect failure because current metadata has no owner.**
- [ ] **Step 3: Change `ingest_text_to_vector_store`, status, clear, and retrieval helpers to require `user_id`; attach it to every chunk and use Chroma filter `{"user_id": user_id}`. Untagged chunks must never be returned or deleted.**
- [ ] **Step 4: Pass the session ID from document/statistics uploads, knowledge-base status/clear, and chat retrieval. Replace global memory-based context with a bounded history loaded by user ID.**
- [ ] **Step 5: Add two-client retrieve/status/clear tests; run `python -m unittest discover -s tests -v`; expect PASS.**
- [ ] **Step 6: Commit with `git add app.py langchain_config.py tests/test_app_contract.py && git commit -m "feat: isolate chat context and knowledge base"`.**

### Task 4: Document configuration and verify

**Files:**

- Modify: `README.md:130-142`
- Test: `tests/test_app_contract.py`

- [ ] **Step 1: Write a failing test showing `SECRET_KEY` is accepted as explicit application configuration.**
- [ ] **Step 2: Add README instructions for `SECRET_KEY`, anonymous-session cookie loss, and the absence of cross-device recovery.**
- [ ] **Step 3: Run `git diff --check` and `python -m unittest discover -s tests -v`; expect no errors and all tests PASS.**
- [ ] **Step 4: Commit with `git add README.md app.py tests/test_app_contract.py && git commit -m "docs: explain anonymous session isolation"`.**

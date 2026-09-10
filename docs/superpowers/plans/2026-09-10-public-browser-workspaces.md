# Public Browser Workspaces Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Execute inline unless the user requests delegation.

**Goal:** Remove Google OIDC while giving every browser immediate access to an isolated workspace.

**Architecture:** Reuse the signed Flask session's opaque `user_id` as the sole workspace owner across SQL, Chroma, and LangGraph. Remove Google account integration, retain legacy data without reassignment, and apply the existing salted-IP quota to everyone.

**Tech Stack:** Flask sessions, SQLite, LangGraph SqliteSaver, LangChain/Chroma, unittest/pytest, server-rendered HTML.

**Approved design:** `docs/superpowers/specs/2026-09-10-public-browser-workspaces-design.md`.

---

## Baseline and file boundaries

Inspected baseline: HEAD `f86f255`. The staged `.gitignore` modification and untracked `.env ` (with a trailing space) belong to the user. Do not read secrets, alter those files, stage them, or include them in a commit. Recheck the baseline before implementation.

Modify `app.py` for session establishment, persistence, agent wiring, quotas, route access, schema creation, and health reporting. Delete `auth.py` after all callers are removed. Modify `templates/chatbot.html`, `.env.example`, `requirements.txt`, and `README.md` for removal and documentation. Update obsolete account-only assertions in `tests/test_app_contract.py`, `tests/test_agent_memory.py`, and `tests/test_auth.py`; rename the latter to `tests/test_browser_sessions.py`. Keep storage-level `user_id` APIs in `langchain_config.py` and existing database columns to avoid a data migration. Its comments may be updated to say workspace rather than account.

The separate FastAPI legal RAG backend in `src/legal_rag` is not the Flask authentication surface and is outside implementation scope. Existing audio functionality is not present as a registered Flask route; do not invent it as part of authentication removal.

### Task 1: Record an isolated baseline

- [ ] Run `git status --short` and `git diff --stat HEAD`; record pre-existing changes in the execution notes.
- [ ] Select the existing Python environment with Flask, LangChain, and pytest installed; check `python -c 'import flask, langchain, pytest'`. If unavailable, report the dependency blocker without claiming tests passed.
- [ ] Before importing app.py in tests, set APP_STATE_DB, AGENT_CHECKPOINT_DB, and RAG_PERSIST_DIR to paths in one fresh temporary directory. Use APP_ENV=testing, LLM_PROVIDER=none, EMBEDDING_PROVIDER=none, LANGSMITH_TRACING=false. Fixtures that override these variables must also use temporary paths and restore the environment/modules afterward. Never run tests against real SQLite or Chroma directories.
- [ ] Run the existing targeted baseline:

```bash
python -m pytest tests/test_auth.py tests/test_app_contract.py tests/test_quota.py tests/test_agent_memory.py tests/test_rag_isolation.py tests/test_vector_store_scope.py tests/test_chat_streaming.py tests/test_health_reporting.py -k 'not Live' -q
```

Record failures before changes. The bare `pytest` command is insufficient because pyproject.toml limits default discovery to `tests/legal_rag`.

### Task 2: Establish one browser identity and server-side drafts

- [ ] In the existing isolated AppContractTests fixture, replace guest-cookie-only/account-draft expectations with this regression test:

```python
def test_browser_drafts_are_private_and_server_side(self):
    first = self.app_module.app.test_client()
    second = self.app_module.app.test_client()
    for client, draft in ((first, "first private draft"), (second, "second private draft")):
        response = client.post("/update_statement", json={"personal_statement_draft": draft})
        self.assertEqual(response.status_code, 200)
    self.assertEqual(first.get("/get_initial_state").json["personal_statement_draft"], "first private draft")
    self.assertEqual(second.get("/get_initial_state").json["personal_statement_draft"], "second private draft")
    with first.session_transaction() as cookie:
        owner = cookie["user_id"]
        self.assertNotIn("guest_draft", cookie)
    with self.app_module.app.app_context():
        self.assertEqual(self.app_module.get_personal_statement_from_db(owner), "first private draft")
```

- [ ] Run `python -m pytest tests/test_app_contract.py -k browser_drafts -q`; expect failure on the old cookie draft behavior.
- [ ] Keep `establish_anonymous_user()` as the single identity creator. After its existing token-generation branch, remove obsolete keys and migrate only an existing cookie draft with INSERT OR IGNORE:

```python
for key in ("authenticated_user_id", "account_email", "auth_next", "oidc_nonce"):
    session.pop(key, None)
legacy_draft = session.pop("guest_draft", None)
if isinstance(legacy_draft, str):
    connection = get_db()
    connection.execute(
        "INSERT OR IGNORE INTO personal_statement (user_id, draft) VALUES (?, ?)",
        (session["user_id"], legacy_draft),
    )
    connection.commit()
```

- [ ] Replace every `auth.get_current_user_id()` in app.py with `session["user_id"]`, rename local `authenticated_user_id` variables to `user_id`, and remove unreachable guest branches. `get_initial_state()` returns the existing two keys using `get_personal_statement_from_db(user_id)` and `get_chat_history_from_db(user_id)`. `update_statement()` calls `set_personal_statement_in_db(user_id, new_draft)` inside its existing try/except. Keep request validation/status codes unchanged.
- [ ] Extend the draft test with a pre-existing SQLite draft plus a different guest_draft in its cookie; verify the SQLite draft wins. A cleared cookie must receive a fresh ID and an empty initial state. Run `python -m pytest tests/test_app_contract.py -k 'draft or anonymous' -q`.

### Task 3: Remove Google integration and open existing routes

- [ ] Replace obsolete OAuth tests with browser-access assertions, retaining the existing production SECRET_KEY and cookie-security subprocess tests:

```python
def test_google_routes_are_removed(self):
    client = self.app_module.app.test_client()
    for method, path in (("get", "/auth/login"), ("get", "/auth/callback"), ("post", "/auth/logout")):
        with self.subTest(path=path):
            self.assertEqual(getattr(client, method)(path).status_code, 404)

def test_chat_page_has_no_google_login(self):
    response = self.app_module.app.test_client().get("/chatbot.html")
    self.assertEqual(response.status_code, 200)
    self.assertNotIn('id="auth-bar"', response.get_data(as_text=True))
    self.assertNotIn("Sign in with Google", response.get_data(as_text=True))
```

- [ ] Run these tests before removing routes; expect failures on the current Google routes/sign-in strip.
- [ ] Delete `import auth`, `auth.init_oauth(app)`, `auth.register_auth_routes(app)`, `app.auth = auth`, the authenticated-account context processor, every `@auth.login_required`, the three GOOGLE_OIDC config assignments, the GOOGLE_OIDC_CONFIGURED computation, and the `google_oidc_configured` health entry. Delete auth.py after its last reference is gone. Do not replace decorators with an always-true login stub.
- [ ] Remove the users CREATE TABLE block from `ensure_database_schema`. Leave all existing tables/rows intact and retain daily_ai_quota's compatible schema and constraints. Update schema tests to check new installs omit users and existing installs preserve a seeded users table across two schema calls.
- [ ] Remove the complete auth-strip comment and `div#auth-bar` from templates/chatbot.html. Keep `div#app`, fonts, scripts, and navigation intact.
- [ ] Update the four previously gated endpoint tests: GET `/knowledge_base/status`, POST `/knowledge_base/clear`, POST `/statistics/analyze`, POST `/process_document_ocr`. Empty upload requests must reach the existing validation response rather than 401/redirect. Use the existing valid mocked upload fixtures for successful requests; assert exact existing success/validation codes, not merely `status != 401`.
- [ ] Run `python -m pytest tests/test_browser_sessions.py tests/test_app_contract.py tests/test_health_reporting.py -q` after renaming tests/test_auth.py and removing its Authlib fakes/import.

### Task 4: Preserve private history, retrieval, and agent memory

- [ ] Adapt AppCheckpointerWiringTests.drive_chat to establish a normal browser session via GET `/health` and read its `user_id`; do not seed authenticated_user_id. Return that ID alongside the recorded agent call. Replace the guest-no-memory assertion with equality between that browser ID, build_agent's user_id, config.configurable.thread_id, and AgentContext.user_id. Drive two distinct clients and assert different thread IDs; drive the same client twice and assert the same ID.
- [ ] Use the following wiring in `_generate_agent_response_stream`, preserving streaming and response parsing:

```python
thread_user_id = session["user_id"]
agent = build_agent(thread_user_id, checkpointer=CHECKPOINTER)
run_config = build_run_config(thread_user_id)
```

Keep `context=AgentContext(user_id=thread_user_id)` on the stream invocation. Persist assistant responses and POSTed user messages to that same owner. Remove the now-obsolete authenticated-only comments and guards.
- [ ] `/get_similar_chats` must execute its existing owner-filtered SQL for every visitor. `/clear_chat_history_endpoint` must call both `clear_chat_history_in_db(user_id)` and `clear_agent_thread(user_id)` for every visitor. Preserve the existing checkpoint-failure response policy; do not add unrelated error semantics.
- [ ] All knowledge-base operations must pass the same `session["user_id"]`: status, clear, search, source deletion, OCR ingestion, and statistics ingestion. Keep the storage-level ownership filter and refuse to accept a submitted owner override.
- [ ] Adapt the existing mocked document upload success test to an anonymous client; assert the ingestion mock's user_id equals the client's session ID. Then exercise status/search/source-delete/clear for both clients against a fake store containing both owners. Assert each mutation affects only its caller and retrieval never returns the other owner's document. Include forged `user_id` JSON/query parameters and verify they do not change ownership.
- [ ] Seed a legacy account draft/history and an old authenticated_user_id cookie. Verify the browser receives an unrelated anonymous ID, cannot see the seeded content, and the original database rows remain. Retain storage-level RAG isolation tests unchanged as an additional layer.
- [ ] Run `python -m pytest tests/test_agent_memory.py tests/test_rag_isolation.py tests/test_vector_store_scope.py tests/test_app_contract.py tests/test_chat_streaming.py -k 'not Live' -q`.

### Task 5: Keep quotas independent of browser identity

- [ ] Replace account-quota route test setup with salted-IP rows; keep concurrent-reservation and UTC rollover tests in tests/test_quota.py. Add a two-client route test with the same REMOTE_ADDR, then a third client with a different REMOTE_ADDR: the first pair shares exhaustion and the third retains allowance. Clearing the first client's cookie must not replenish its allowance.
- [ ] Replace reserve_ai_request() with:

```python
def reserve_ai_request():
    """Reserve one daily AI request for the direct peer's salted IP identity."""
    client_ip = client_ip_from_request(request)
    return reserve_daily_quota(
        get_db(),
        "guest_ip",
        guest_ip_identity(client_ip, app.config["QUOTA_IP_SALT"]),
        limit=app.config["DAILY_AI_REQUEST_LIMIT"],
    )
```

- [ ] Verify exhausted chat emits the existing SSE error without calling the agent; exhausted OCR/statistics returns the existing JSON 429 without calling the paid/model processing step. Failed extraction/invalid uploads do not reserve quota. Preserve direct-peer handling and the test that ignores forged X-Forwarded-For.
- [ ] Run `python -m pytest tests/test_quota.py tests/test_app_contract.py -q`. Keep quota.py's legacy `user` support for database/API compatibility; the Flask app must only reserve guest_ip entries.

### Task 6: Configuration, documentation, and final verification

- [ ] Remove only `Authlib==1.8.0` from requirements.txt and the three GOOGLE_OIDC entries from .env.example. Keep unrelated Google provider/API configuration. Do not touch either private .env file.
- [ ] Replace README's Google setup instructions and authentication architecture labels with browser-workspace behavior. State: no login; cookie continuity only; clearing cookies does not delete server data; old account data is preserved but inaccessible through anonymous workspaces; users sharing an IP share the configured daily allowance; SECRET_KEY remains required in production. Keep README's existing Korean language and accurately distinguish this Flask app from the independent legal RAG backend.
- [ ] Search runtime/config/test files for stale integration references:

```bash
rg -n 'GOOGLE_OIDC|authlib|Authlib|auth\.|authenticated_user_id|login_required|auth_login|auth_logout' app.py langchain_config.py templates static requirements.txt .env.example tests
```

Expected matches are only deliberate obsolete-cookie cleanup and negative regression tests. Historical specs/plans stay unchanged.
- [ ] Run the complete targeted command from Task 1 with tests/test_browser_sessions.py substituted for tests/test_auth.py. Then run `python -m pytest tests --ignore=tests/legal_rag/live --ignore=tests/test_ollama_connector_live.py -k 'not Live' -q` with the same temporary storage environment. Report baseline failures separately; do not silently accept new failures.
- [ ] With a local test server and mocked/disabled model providers, open the chat page in two independent browser sessions. Confirm the sign-in strip is absent, the app renders, drafts remain separate after reload, and clearing one history leaves the other untouched. Record any unavailable browser verification explicitly.
- [ ] Run `git diff --check` and inspect the final diff. Stage only the implementation's explicit file paths if committing; never `git add .`, and never commit the user's already-staged .gitignore change inadvertently. Use a path-limited commit or leave changes uncommitted if selective staging cannot be performed safely.

## Completion evidence

Report changed files, exact test commands/results, verified browser behavior, and remaining limitations. No live model test is needed for identity changes. Do not claim audio support was enabled, deployment was secured, abandoned workspaces were deleted, or legacy account records were migrated. The task is complete only when all existing public feature paths use one private browser identity and the isolation/quota regressions pass.

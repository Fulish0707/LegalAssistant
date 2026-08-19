# Google OIDC User Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Google OIDC accounts, account-private state and RAG, and a 20-request daily AI quota while preserving a limited non-persistent guest experience.

**Architecture:** Keep Flask as the web boundary, but move identity and quota logic into focused modules. `auth.py` owns Google OIDC and the request identity; `quota.py` owns atomic daily quota reservations; `langchain_config.py` builds a filtered retriever on demand rather than mutating global request state. SQLite is the source of truth for accounts and quotas; Chroma chunks carry the internal account ID and a document ID.

**Tech Stack:** Flask, Authlib OAuth/OIDC client, SQLite, Chroma, LangChain, Python `unittest`.

---

## File structure

| File | Responsibility |
| --- | --- |
| `app.py` | Flask configuration, schema migration, existing routes, guest/session behavior, and dependency wiring. |
| `auth.py` | OIDC client initialization, sign-in/callback/logout routes, identity lookup, and the API/page authentication guard. |
| `quota.py` | UTC-day quota identity derivation and atomic SQLite reservation. |
| `langchain_config.py` | User-filtered retriever construction, user-owned document metadata, and request-specific RAG tool creation. |
| `requirements.txt` | Authlib runtime dependency. |
| `.env.example` | Non-secret Google OIDC and quota configuration names. |
| `templates/chatbot.html` | Sign-in/sign-out state and guest capability messaging. |
| `tests/test_auth.py` | Mocked OIDC, session, authorization, and guest persistence tests. |
| `tests/test_quota.py` | SQLite quota boundary and isolation tests. |
| `tests/test_rag_isolation.py` | Request-scoped retrieval and document ownership tests. |
| `tests/test_app_contract.py` | Route authorization and updated authenticated upload contracts. |

### Task 1: Add configuration and schema migration

**Files:**
- Modify: `requirements.txt:1-25`
- Modify: `.env.example`
- Modify: `app.py:58-100`
- Test: `tests/test_auth.py`

- [ ] **Step 1: Write failing configuration/schema tests**

```python
def test_schema_creates_users_and_daily_ai_quota_tables(self):
    conn = sqlite3.connect(self.app_module.DB_FILE)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    self.assertTrue({"users", "daily_ai_quota"}.issubset(tables))

def test_missing_google_oidc_settings_leave_login_disabled(self):
    response = self.client.get("/auth/login")
    self.assertEqual(response.status_code, 503)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m unittest tests.test_auth -v`

Expected: failures because the tables and `/auth/login` route do not exist.

- [ ] **Step 3: Add the runtime settings and dependency**

Add `Authlib` to `requirements.txt`. Add only names and non-secret examples to `.env.example`:

```dotenv
GOOGLE_OIDC_CLIENT_ID=
GOOGLE_OIDC_CLIENT_SECRET=
GOOGLE_OIDC_REDIRECT_URI=http://localhost:5005/auth/callback
QUOTA_IP_SALT=replace-with-a-long-random-secret
TRUSTED_PROXY_COUNT=0
```

Configure Flask session cookies in `app.py` from `FLASK_ENV`/`SESSION_COOKIE_SECURE`, with `HttpOnly=True`, `SameSite="Lax"`, and secure cookies enabled outside development. Do not generate a random production `SECRET_KEY`; fail startup in non-development environments when it is absent.

- [ ] **Step 4: Extend `ensure_database_schema` idempotently**

Create these tables and indexes without changing legacy-data ownership:

```sql
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  google_subject TEXT NOT NULL UNIQUE,
  email TEXT,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_login_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS daily_ai_quota (
  identity_kind TEXT NOT NULL CHECK(identity_kind IN ('user', 'guest_ip')),
  identity_value TEXT NOT NULL,
  quota_day TEXT NOT NULL,
  request_count INTEGER NOT NULL DEFAULT 0 CHECK(request_count >= 0),
  PRIMARY KEY (identity_kind, identity_value, quota_day)
);
CREATE INDEX IF NOT EXISTS idx_chat_history_user_id_id ON chat_history(user_id, id);
```

Use `uuid.uuid4().hex` for internal IDs. Leave `legacy-unassigned` rows untouched and inaccessible.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m unittest tests.test_auth -v`

Expected: the schema test passes; the login test may remain failing until Task 2.

- [ ] **Step 6: Commit the schema/configuration slice**

```bash
git add requirements.txt .env.example app.py tests/test_auth.py
git commit -m "feat: add account and quota persistence schema"
```

### Task 2: Implement Google OIDC and current-user boundaries

**Files:**
- Create: `auth.py`
- Modify: `app.py:1-100`
- Modify: `templates/chatbot.html`
- Test: `tests/test_auth.py`

- [ ] **Step 1: Write failing OIDC and authorization tests**

```python
def test_callback_creates_account_and_records_internal_user_id(self):
    self.app_module.oauth.google.authorize_access_token = lambda: {"access_token": "test"}
    self.app_module.oauth.google.parse_id_token = lambda token, nonce=None: {
        "sub": "google-subject-1", "email": "person@example.com"
    }
    response = self.client.get("/auth/callback", follow_redirects=False)
    self.assertEqual(response.status_code, 302)
    with self.client.session_transaction() as saved:
        self.assertTrue(saved["authenticated_user_id"])

def test_account_only_api_returns_401_for_guest(self):
    response = self.client.get("/knowledge_base/status")
    self.assertEqual(response.status_code, 401)
    self.assertEqual(response.json["error"], "authentication_required")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m unittest tests.test_auth.AppAuthTests.test_callback_creates_account_and_records_internal_user_id tests.test_auth.AppAuthTests.test_account_only_api_returns_401_for_guest -v`

Expected: import/route failures.

- [ ] **Step 3: Create `auth.py` with explicit interfaces**

Implement `init_oauth(app)`, `get_current_user_id()`, `login_required(view)`, `register_auth_routes(app)`, and `upsert_google_user(subject, email)`. Register Google with discovery URL `https://accounts.google.com/.well-known/openid-configuration`, scope `openid email profile`, and a per-login nonce. Callback behavior must require a non-empty `sub`, upsert by `google_subject`, store `authenticated_user_id`, remove `guest_draft`, and redirect to `/`.

`login_required` returns this JSON response for API routes:

```python
return jsonify({"error": "authentication_required"}), 401
```

For HTML routes, redirect to `/auth/login?next=<relative-path>` and reject external `next` values.

- [ ] **Step 4: Wire user interface and logout behavior**

Render current account email or a Google sign-in link in `chatbot.html`; never place an access token, ID token, or Google subject in the DOM. Implement `POST /auth/logout` to clear account identity and guest draft, then redirect to `/`.

- [ ] **Step 5: Run authentication tests**

Run: `python -m unittest tests.test_auth -v`

Expected: mocked sign-in, repeat login/upsert, invalid callback, logout, and guest authorization tests pass.

- [ ] **Step 6: Commit the authentication slice**

```bash
git add auth.py app.py templates/chatbot.html tests/test_auth.py
git commit -m "feat: add Google OIDC account sessions"
```

### Task 3: Separate account persistence from guest session state

**Files:**
- Modify: `app.py:127-160,438-530`
- Test: `tests/test_auth.py`
- Test: `tests/test_app_contract.py`

- [ ] **Step 1: Write failing guest/account persistence tests**

```python
def test_guest_draft_is_session_only(self):
    guest = self.app_module.app.test_client()
    guest.post("/update_statement", json={"personal_statement_draft": "temporary"})
    self.assertEqual(guest.get("/get_initial_state").json["personal_statement_draft"], "temporary")
    guest.delete_cookie("session")
    self.assertEqual(guest.get("/get_initial_state").json["personal_statement_draft"], "")

def test_two_signed_in_users_cannot_read_each_others_drafts_or_chat(self):
    first = self.app_module.app.test_client()
    second = self.app_module.app.test_client()
    with first.session_transaction() as saved:
        saved["authenticated_user_id"] = "account-one"
    with second.session_transaction() as saved:
        saved["authenticated_user_id"] = "account-two"
    first.post("/update_statement", json={"personal_statement_draft": "first draft"})
    second.post("/update_statement", json={"personal_statement_draft": "second draft"})
    self.assertEqual(first.get("/get_initial_state").json["personal_statement_draft"], "first draft")
    self.assertEqual(second.get("/get_initial_state").json["personal_statement_draft"], "second draft")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m unittest tests.test_auth.AppAuthTests.test_guest_draft_is_session_only tests.test_app_contract.AppContractTests.test_drafts_are_session_scoped -v`

Expected: failure because guests currently write drafts/chat to SQLite.

- [ ] **Step 3: Route persistence through the current identity**

Use `authenticated_user_id` only for `get_personal_statement_from_db`, `set_personal_statement_in_db`, all chat-history helpers, and RAG APIs. For guests, store only `session["guest_draft"]`; return an empty `chat_history`; do not call SQLite chat helper functions. Update initial state and update/clear routes to preserve this contract.

- [ ] **Step 4: Restrict account-only product APIs**

Decorate document OCR upload, statistics upload, knowledge-base status, knowledge-base clear, and any audio endpoint that invokes an AI provider with `login_required`. Keep guest chat and draft updates available. Make API response tests assert `401` before file processing.

- [ ] **Step 5: Run persistence and route contracts**

Run: `python -m unittest tests.test_auth tests.test_app_contract -v`

Expected: guests retain only session drafts; accounts retain private logs/drafts; account-only APIs return `401` for guests.

- [ ] **Step 6: Commit the persistence boundary**

```bash
git add app.py tests/test_auth.py tests/test_app_contract.py
git commit -m "feat: separate guest and account workspace state"
```

### Task 4: Add atomic 20-request daily quota enforcement

**Files:**
- Create: `quota.py`
- Modify: `app.py:310-405,560-725`
- Test: `tests/test_quota.py`

- [ ] **Step 1: Write failing quota unit tests**

```python
def test_twenty_reservations_succeed_and_twenty_first_is_rejected(self):
    for _ in range(20):
        self.assertTrue(reserve_daily_request(self.conn, "user", "u1").allowed)
    result = reserve_daily_request(self.conn, "user", "u1")
    self.assertFalse(result.allowed)
    self.assertEqual(result.limit, 20)

def test_user_and_hashed_guest_ip_have_independent_buckets(self):
    self.assertTrue(reserve_daily_request(self.conn, "user", "u1").allowed)
    self.assertTrue(reserve_daily_request(self.conn, "guest_ip", "hashed-ip").allowed)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_quota -v`

Expected: module import failure.

- [ ] **Step 3: Implement `quota.py`**

Create `QuotaResult(allowed, used, limit, quota_day)` and `reserve_daily_request(conn, identity_kind, identity_value, now=None, limit=20)`. Use an SQLite transaction with an `INSERT` that has an `ON CONFLICT` clause to increment `request_count` only when it is less than 20, then read the row to distinguish a rejected request. Use `datetime.now(timezone.utc).date().isoformat()`.

Create `get_quota_identity(request, user_id, salt, trusted_proxy_count)`: use `("user", user_id)` for authenticated users; otherwise normalize the trusted client IP and return `("guest_ip", sha256(f"{salt}:{ip}".encode()).hexdigest())`. Do not persist raw IP values.

- [ ] **Step 4: Enforce reservations immediately before every model invocation**

Wrap chat generation, OCR `analyze_document`, statistics `llm.invoke`, and AI-backed STT paths with a shared `require_quota()` helper. Return `429` JSON for normal APIs and an SSE `error` then `stream_end` for chat. Do not reserve quota for validation errors, failed authentication, local-only statistics, or status routes.

- [ ] **Step 5: Run quota tests and app contracts**

Run: `python -m unittest tests.test_quota tests.test_app_contract -v`

Expected: 20 accepted/21st rejected behavior, daily/identity separation, and correct `429` response shape.

- [ ] **Step 6: Commit quota enforcement**

```bash
git add quota.py app.py tests/test_quota.py tests/test_app_contract.py
git commit -m "feat: enforce daily AI request quotas"
```

### Task 5: Eliminate process-global RAG request state

**Files:**
- Modify: `langchain_config.py:90-300`
- Modify: `app.py:1-15,310-405,520-540,610-700`
- Test: `tests/test_rag_isolation.py`

- [ ] **Step 1: Write failing RAG isolation tests**

```python
def test_retrievers_are_created_with_the_callers_user_filter(self):
    captured_filters = []
    fake_store = FakeVectorStore(captured_filters)
    self.module.vector_store = fake_store
    self.module.retrieve_for_user("question", "user-a")
    self.module.retrieve_for_user("question", "user-b")
    self.assertEqual(captured_filters, [{"user_id": "user-a"}, {"user_id": "user-b"}])

def test_ingestion_adds_owner_and_document_id_metadata(self):
    result = self.module.ingest_text_to_vector_store("text", "a.txt", user_id="user-a")
    self.assertTrue(result["document_id"])
    self.assertEqual(self.fake_store.last_metadata["user_id"], "user-a")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m unittest tests.test_rag_isolation -v`

Expected: failure because `retriever` is a mutable module-global value.

- [ ] **Step 3: Implement request-specific retrieval**

Remove the exported global `retriever` and unused global `memory`. Add:

```python
def get_user_retriever(user_id):
    if vector_store is None:
        return None
    return vector_store.as_retriever(search_kwargs={"k": 3, "filter": {"user_id": user_id}})

def retrieve_for_user(query, user_id):
    retriever = get_user_retriever(user_id)
    return retriever.invoke(query) if retriever else []
```

Change RAG tool construction to `build_rag_tool(user_id)` and build the agent/request tool set with the authenticated user ID. Do not mutate module-level retriever state during ingestion, clearing, or retrieval.

- [ ] **Step 4: Add ownership metadata and preserve account-only clear**

Generate `document_id = uuid.uuid4().hex` once per ingestion and add `document_id`, `user_id`, `source`, and UTC `ingested_at` metadata to every chunk. Return it in the ingestion result. Preserve `where={"user_id": user_id}` for status and clear; ensure an empty clear never changes another account's data.

- [ ] **Step 5: Run RAG isolation tests**

Run: `python -m unittest tests.test_rag_isolation tests.test_app_contract -v`

Expected: sequential and concurrent simulated users have independent Chroma filters; metadata contains ownership/document identity.

- [ ] **Step 6: Commit request-scoped RAG**

```bash
git add langchain_config.py app.py tests/test_rag_isolation.py tests/test_app_contract.py
git commit -m "fix: scope RAG retrieval to authenticated users"
```

### Task 6: Validate the complete foundation and update operator documentation

**Files:**
- Modify: `README.md:90-125`
- Modify: `docs/integration-roadmap.md`
- Modify: `docs/superpowers/specs/2026-08-19-google-oidc-user-foundation-design.md` only if implementation reveals an approved design correction
- Test: `tests/test_auth.py`
- Test: `tests/test_quota.py`
- Test: `tests/test_rag_isolation.py`
- Test: `tests/test_app_contract.py`

- [ ] **Step 1: Write a final health/configuration contract**

```python
def test_health_exposes_oidc_readiness_without_secrets(self):
    response = self.client.get("/health")
    self.assertIn("google_oidc_configured", response.json)
    self.assertNotIn("GOOGLE_OIDC_CLIENT_SECRET", response.get_data(as_text=True))
```

- [ ] **Step 2: Document Google setup and guest policy**

Update README setup instructions with Google Cloud OAuth consent screen, authorised redirect URI, required environment variable names, and the 20-request/day guest/account policy. State that production needs HTTPS and a persistent `SECRET_KEY`; do not put secret values in the repository.

- [ ] **Step 3: Mark roadmap progress**

Change Priority 1 in `docs/integration-roadmap.md` from `Planned` to `Integrated` only after all final tests pass. Keep later items unchanged.

- [ ] **Step 4: Run the full test suite**

Run: `python -m unittest discover -s tests -v`

Expected: all contract, sanitizer, authentication, quota, and RAG-isolation tests pass.

- [ ] **Step 5: Run static safety checks**

Run: `git diff --check`

Expected: no whitespace errors.

Run: `rg -n "GOOGLE_OIDC_CLIENT_SECRET|access_token|id_token" README.md templates tests`

Expected: no secret value, token logging, or token rendering path; environment variable names in README are allowed.

- [ ] **Step 6: Commit documentation and verification tests**

```bash
git add README.md docs/integration-roadmap.md tests/test_auth.py tests/test_quota.py tests/test_rag_isolation.py
git commit -m "docs: document Google account integration"
```

## Plan self-review

- Every requirement in the approved design maps to a task: OIDC (Task 2), data boundary (Task 3), quota (Task 4), RAG isolation (Task 5), and configuration/docs (Tasks 1 and 6).
- Guest policy, account policy, lack of guest migration, 20/day limit, any-Google-account access, and no subscription tier are explicit.
- This plan intentionally leaves upload hardening, streaming redesign, document-level deletion, deployment, dependency pinning, CI, and STT repair for their separately tracked stages.

# Demo Prototype Integration — TDD Checklist

**Goal:** Finish a functional Google-sign-in demo with temporary guest chat/text use, account-private persistent data and RAG, and a simple 20-request daily AI limit.

**Demo boundary:** Guest quota identity is always `request.remote_addr` salted and hashed. Proxy-header interpretation, subscription billing, production observability, and deployment hardening are deferred.

## Integration rule

For every item below: write the named failing test first, run it and record the expected failure, write only the minimum implementation, re-run the focused test, run the bounded full suite, then commit. No later item begins until its specification and code-quality reviews are approved.

## Current verified work

- [x] Lazy-load microphone libraries so application import and tests do not hang.
- [x] Add secure-by-default configuration and account/quota SQLite schema.
- [x] Add Google OIDC sign-in, logout, safe callback handling, and atomic account upsert.
- [x] Separate guest session state from account-persistent drafts/chat and protect upload/knowledge-base APIs.
- [x] Add a 20/day quota for AI calls; use account ID for signed-in users and salted `remote_addr` for guests.
- [ ] Complete quota code-quality review after the simplified IP change.

## Remaining implementation tasks

### 1. Finalize quota review

- [ ] Inspect `quota.py`, all AI invocation sites, and quota tests against the demo boundary.
- [ ] Confirm the guest identity test proves `X-Forwarded-For` is ignored and only `request.remote_addr` affects the hash.
- [ ] Confirm 20 accepted reservations and the 21st rejection for both a guest and an account; confirm separate identity/day buckets.
- [ ] Confirm chat emits SSE `error` then `stream_end`, while normal AI APIs return JSON `429`.
- [ ] Run: `timeout 60s python -m unittest discover -s tests -v`.
- [ ] Run: `git diff --check`.

### 2. Replace the mutable global RAG retriever (TDD)

- [ ] Write `tests/test_rag_isolation.py::test_retriever_uses_callers_account_filter` with a fake Chroma store. Invoke retrieval for `account-a` and `account-b`; assert filters are exactly `{"user_id": "account-a"}` and `{"user_id": "account-b"}`.
- [ ] Run that test; expected failure: retrieval relies on the module-global `retriever` or on the most recent ingestion.
- [ ] Write `test_ingestion_attaches_owner_and_document_id` asserting each chunk has account `user_id`, one generated `document_id`, source, and UTC ingestion time.
- [ ] Run it; expected failure: no document ID/timestamp contract.
- [ ] Implement `get_user_retriever(user_id)`, `retrieve_for_user(query, user_id)`, and `build_rag_tool(user_id)` in `langchain_config.py`. Remove request mutation of the global retriever and unused global memory from application flow.
- [ ] Change agent creation/call flow in `app.py` to construct the RAG tool with the current authenticated ID; do not enable RAG for guests.
- [ ] Keep status/clear queries filtered by authenticated account ID.
- [ ] Run focused RAG tests, then `timeout 60s python -m unittest discover -s tests -v`, then `git diff --check`.
- [ ] Commit: `fix: scope RAG retrieval to signed-in accounts`.

### 3. Demo acceptance tests (TDD)

- [ ] Add a single happy-path account test: mocked Google callback → account draft saved → account upload ingests chunks tagged with that account → account status lists only those chunks.
- [ ] Add a guest-flow test: guest chat/text draft succeeds; guest upload and knowledge-base requests return exact `401`; a new guest session has no prior draft/chat.
- [ ] Add a two-account isolation test: account A and B each ingest mocked content; retrieval/status/clear for A never sees or deletes B's records.
- [ ] Add a quota-flow test: consume 20 mocked chat reservations; verify the 21st produces SSE quota error and no LLM invocation.
- [ ] Run each test red before its supporting integration adjustment, then run all demo acceptance tests green.
- [ ] Run: `timeout 60s python -m unittest discover -s tests -v`.

### 4. Demo operator documentation

- [ ] Update `README.md` with local setup: `SECRET_KEY`, `GOOGLE_OIDC_CLIENT_ID`, `GOOGLE_OIDC_CLIENT_SECRET`, `GOOGLE_OIDC_REDIRECT_URI`, and `QUOTA_IP_SALT`.
- [ ] State the demo policies exactly: any Google account can sign in; guest chat/text is temporary; account logs/drafts/knowledge base are private; both tiers have 20 AI requests/day; guest quota uses direct remote IP only.
- [ ] Add a manual demo script: open guest workspace, attempt upload (401), sign in with Google, save draft/upload document, ask document question, exhaust a test quota, sign out.
- [ ] Update `docs/integration-roadmap.md`: mark quota integrated only after its review; mark private request-scoped RAG integrated only after Task 2 passes.
- [ ] Run final full suite and `git diff --check`; commit documentation separately.

## Explicitly deferred

- Trusted-proxy/X-Forwarded-For support, rate-limit distribution, billing/subscriptions, production WSGI/deployment, file-hardening limits, citations/document deletion UI, CI/dependency pinning, app modularization, and the STT keyword bug.

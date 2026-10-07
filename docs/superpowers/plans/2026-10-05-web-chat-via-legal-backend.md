# Web Chat via legal_rag Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Every web chat turn is answered by legal_rag `/v1/chat` (configurable generator + verifier) instead of the Flask agent.

**Architecture:** A new root module `legal_backend.py` posts the message to `/v1/chat` and renders the structured response into chat text and source cards. `app.py`'s SSE generator for `/chat` calls it and emits the same event types the front end already consumes. No fallback to the local agent.

**Tech Stack:** Flask, httpx, unittest (run with pytest), legal_rag FastAPI service.

Design: `docs/superpowers/specs/2026-10-05-web-chat-via-legal-backend-design.md`

Test command (never touches real storage or a live model):

```
T=$(mktemp -d); env APP_STATE_DB=$T/app_state.db AGENT_CHECKPOINT_DB=$T/ckpt.db RAG_PERSIST_DIR=$T/chroma \
  APP_ENV=testing LLM_PROVIDER=none EMBEDDING_PROVIDER=none LANGSMITH_TRACING=false \
  python -m pytest tests --ignore=tests/legal_rag --ignore=tests/test_ollama_connector_live.py -k 'not Live' -q
```

Record the baseline pass/fail counts before Task 1.

---

### Task 1: `legal_backend.py` — call and render

**Files:**
- Create: `legal_backend.py`
- Test: `tests/test_legal_backend.py`

**Behaviour to test (write these first, watch them fail):**

1. `ask(base_url, message, conversation_id)` posts `{"message", "conversation_id"}` to `<base_url>/v1/chat` (trailing slash on the base tolerated) and returns the JSON body.
2. A connection error, a timeout, a non-2xx status and a non-JSON body each raise `LegalBackendUnavailable`.
3. `render(body)` for `kind: answer` returns `(text, sources)`:
   - text has the summary first, then the analysis, then labelled lists `취할 수 있는 조치`, `추가로 확인이 필요한 사실`, and a `불확실한 점` paragraph; empty sections are omitted;
   - bare source ids in the prose (`(S1, S2)`, `S3`) are rewritten to `[1]`, `[2]`, matching the numbering of the sources panel;
   - sources is one `{source, quote, kind, score}` per evidence item, in order; `kind` is `법령` for statute evidence and `판례` otherwise; `source` joins the header values; `quote` is the evidence text capped at 400 characters.
4. `render` for `kind: clarification` lists the backend's questions under a line asking for more detail; sources empty.
5. `render` for `kind: rejected` returns the fixed "근거를 확인할 수 없어" message; sources empty.

**Implementation notes:**
- `TIMEOUT = 180.0` seconds: a turn with one regeneration measured up to 45 s; the first request also loads the reranker.
- Evidence headers are a dict of display strings (`Law`, `Article`, `Court`, `Case number`, ...); join the values with spaces, de-duplicated, as `_legal_hit_source` does.
- The module imports only `httpx` and `re`; it must stay importable without LangChain.

### Task 2: route `/chat` through the backend

**Files:**
- Modify: `app.py` (`_generate_agent_response_stream`, the `/chat` GET branch)
- Test: `tests/test_chat_streaming.py`, `tests/test_forced_legal_retrieval.py`, `tests/test_app_contract.py`

**New generator `_generate_legal_response_stream(user_input)`:**

1. `LEGAL_RAG_URL` unset → `error` event "법률 답변 서비스가 설정되지 않았습니다. LEGAL_RAG_URL을 설정한 뒤 다시 시작하세요." then `stream_end`.
2. Reserve quota exactly as today; exhaustion emits the same `daily_ai_quota_exceeded` error.
3. Greeting or thanks (`_needs_legal_search` false) → `final_response` with a fixed greeting, no backend call.
4. Otherwise emit `thought` "법령·판례 검색 및 답변 작성 중", call `legal_backend.ask(url, user_input, session["user_id"])`, render, run `annotate_unsupported(text, unsupported_citations(text, evidence texts + [user_input]))`, store the assistant message, emit `final_response` with `chat_response`, `sources`, and the three suggestion fields empty.
5. `LegalBackendUnavailable` → log, store and emit `error` "법률 답변 서비스를 사용할 수 없습니다. 잠시 후 다시 시도해 주세요."
6. Always end with `stream_end`.

The `/chat` GET branch returns this generator. `personal_statement_draft` is still accepted in the query string and ignored.

**Removed:** `_generate_agent_response_stream`. Once unrouted it cannot be exercised through `/chat`, so it and the tests that drive it through `/chat` go together. Helpers it used (`_AnswerTokenFilter`, `_tool_sources`, `build_agent`, ...) stay for the separate cleanup, with their direct tests.

**Tests (stub `legal_backend.ask` with `mock.patch.object(app_module.legal_backend, "ask", ...)`):**

- answer → events are `thought`, `final_response`, `stream_end`, all in `FRONTEND_EVENT_TYPES`; `final_response.sources` matches the evidence; the three suggestion fields are `""`.
- the backend is called with the browser's session id as conversation id.
- greeting → no backend call, a `final_response`.
- clarification and rejected bodies → `final_response` with the rendered text.
- `LegalBackendUnavailable` → `error` then `stream_end`; no `final_response`.
- no `LEGAL_RAG_URL` → configuration `error`.
- the assistant reply is stored in chat history.
- missing message still yields `error`, `stream_end` (existing test, unchanged).
- quota tests in `test_app_contract.py`: stub `ask` and count calls instead of patching `build_agent`; `test_chat_sse_reports_missing_ai_configuration` becomes the no-`LEGAL_RAG_URL` case.

Delete: agent-streaming and draft-context tests in `test_chat_streaming.py` and `ForcedRetrievalStreamTests` in `test_forced_legal_retrieval.py`. Keep `AnswerTokenFilterTests` and `LegalEvidenceMiddlewareTests` (they test helpers directly).

### Task 3: configuration and docs

**Files:**
- Modify: `.env` comment block above `LLM_PROVIDER` and above `LEGAL_RAG__LLM__BACKEND` (comments only; do not change the user's values)
- Modify: `README.md` where the chat flow is described

State that the web chat answers through `LEGAL_RAG_URL` `/v1/chat`, that the generator is chosen by `LEGAL_RAG__LLM__BACKEND` / `MODEL` and needs an API restart, and that `LLM_PROVIDER` no longer affects chat.

### Task 4: verify

1. Run the test command; compare with the baseline. Every failure must be pre-existing or explained.
2. Live check, one process at a time and only after the evaluation's judge has finished: start `scripts/serve.py --profile local`, start the Flask app, send one legal question and one greeting to `/chat`, confirm the SSE events and that the reply names its sources. Stop both servers.

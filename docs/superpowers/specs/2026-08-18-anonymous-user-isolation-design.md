# Anonymous User Isolation Design

## Goal

Ensure that each browser session can access only its own personal-statement draft, chat history, in-memory conversation context, and uploaded knowledge-base content without requiring registration or login.

## Scope

This change introduces anonymous, browser-scoped identities. It does not add user accounts, cross-device recovery, sharing, roles, or a general authentication system.

## Identity Model

Flask's signed session cookie stores one opaque `user_id` value. A request that has no `user_id` receives a new value created with `secrets.token_urlsafe(32)`. The cookie is signed with Flask's `SECRET_KEY`; production configuration must provide a stable secret through the environment so users retain their identity across restarts.

Clearing the browser's cookies creates a new anonymous identity. Because no login exists, the prior identity and its data cannot be recovered. The identity is never accepted from a query parameter, request body, or client-side storage.

## SQLite Data Model

`personal_statement` becomes one row per `user_id`:

```sql
CREATE TABLE IF NOT EXISTS personal_statement (
    user_id TEXT PRIMARY KEY,
    draft TEXT NOT NULL DEFAULT ''
)
```

`chat_history` receives a non-null `user_id` column and an index on `(user_id, id)`:

```sql
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_chat_history_user_id_id
    ON chat_history(user_id, id);
```

The startup migration handles the legacy schema safely. It renames the old tables, creates the new tables, and migrates legacy content under a reserved `legacy-unassigned` owner. New anonymous sessions cannot obtain that owner and therefore cannot view historical shared data.

Every draft and chat-history read, write, search, and clear query receives the active session's `user_id` explicitly. The application must never rely on a module-level current-user value.

## Chat Context

The existing module-global LangChain `ConversationBufferMemory` is shared state and is removed from request handling. Before invoking the agent, the app reads the active user's chat rows from SQLite and supplies that bounded history as input. The implementation retains a practical recent-history cap so an unbounded database cannot inflate model prompts.

This makes each request's LLM context a deterministic function of the active session's stored messages and avoids cross-user in-memory contamination.

## Knowledge Base

Every uploaded document and statistics analysis receives `user_id` metadata when ingested into Chroma. The retriever used for a request is created with a metadata filter matching the active `user_id`.

Knowledge-base status and clear operations likewise filter or delete only chunks whose metadata matches the active user. Existing untagged Chroma chunks remain inaccessible rather than being exposed to the first anonymous visitor.

The application's built-in source-folder ingestion is not converted into anonymous user content in this change. It remains a separate future design decision; this scope guarantees that user uploads are isolated.

## Request Flow

1. A Flask `before_request` hook establishes the signed-session `user_id`.
2. Each route passes that ID to storage and retrieval functions.
3. Upload ingestion writes the ID to Chroma chunk metadata.
4. Chat generation retrieves only the active session's rows and chunks.
5. Clear endpoints remove only resources owned by the active session.

## Errors and Configuration

The app uses `SECRET_KEY` from the environment. In development only, a generated process-local fallback permits startup; production deployment must fail fast or be explicitly configured with a stable key. Database migrations execute transactionally. Migration failure must stop startup rather than leave tables in a partially converted state.

## Tests

Tests will create two independent Flask test clients. They will verify that one client cannot:

- read or overwrite the other client's draft;
- read, search, or clear the other client's chat rows;
- retrieve, list, or clear the other client's knowledge-base chunks;
- inherit the other client's chat context during agent invocation.

Migration tests will begin with the legacy schema and verify legacy content is assigned to `legacy-unassigned`, with no ordinary session able to read it.

## Acceptance Criteria

- A new browser has an opaque signed-session identity without a login screen.
- Separate browser sessions observe independent drafts and chat histories.
- Each session's LLM context contains only that session's messages.
- User-uploaded RAG content is only retrievable, visible, and clearable by the creating session.
- Legacy shared database and untagged Chroma data are not exposed to anonymous sessions.
- Existing contract tests remain green and new isolation/migration tests pass.

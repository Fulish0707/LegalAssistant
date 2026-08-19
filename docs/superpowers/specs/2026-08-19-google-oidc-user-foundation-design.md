# Google OIDC User Foundation Design

## Goal

Introduce secure account-scoped application state with Google OpenID Connect (OIDC), while retaining a deliberately limited, non-persistent guest experience.

## Scope and policy

- Any valid Google account may sign in; no email domain or allowlist restriction applies.
- Signed-in users receive persistent personal drafts, chat logs, and a private knowledge base.
- Guests may use chat and the personal-statement text workspace, but have no persistent chat history, uploads, statistics analysis, or knowledge-base access.
- Both guests and signed-in users are limited to 20 AI-generating requests per calendar day (UTC). Subscription tiers are explicitly out of scope.
- Guest data is never migrated when a user signs in. Accounts begin with an empty account-scoped workspace.

## Authentication and identity

The Flask application will use Authlib's Google OIDC client through Google's discovery document. Successful sign-in validates the OIDC response and obtains the stable `sub` claim. The application must never use email as its primary identity key because email can change.

The database gains a `users` table with an internal immutable ID and a unique Google subject value. The logged-in session stores only the internal user ID and an authentication marker. Logout clears those account session fields and rotates/clears guest state.

All account-only routes use one shared `login_required` decorator. It returns JSON `401` for API requests and redirects browser page requests to the sign-in flow. OAuth state and nonce validation use the existing Flask signed session, backed by a production-configured `SECRET_KEY`.

## Data boundaries

`personal_statement`, `chat_history`, and Chroma document metadata are scoped by internal user ID. Existing `legacy-unassigned` rows remain inaccessible, rather than being assigned to a new user.

Guest drafts live only in the signed Flask session. Guest chat entries are not written to SQLite. The existing module-level LangChain memory is removed from request handling; it is neither a user boundary nor safe under concurrent requests.

RAG retrieval is constructed for each request with a `user_id` metadata filter. No mutable global retriever may be updated based on the most recent request. Upload-derived document chunks include a document ID, user ID, source name, and ingestion timestamp to establish a durable ownership boundary for the later RAG lifecycle work.

## Authorization matrix

| Capability | Guest | Signed-in user |
| --- | --- | --- |
| Chat | Yes, no saved history | Yes, saved history |
| Text workspace | Yes, session-only | Yes, persistent |
| OCR/document upload | No | Yes |
| Statistics upload | No | Yes |
| Knowledge-base retrieval | No | Yes, own documents only |
| Knowledge-base ingestion/clear | No | Yes, own documents only |

## Daily quota

Every LLM or local-model invocation is preceded by one quota check in a shared service. A successful reservation consumes one request; rejected requests receive JSON `429` with the applicable daily limit and reset boundary. Failed authentication, invalid uploads, and locally computed non-AI operations do not consume quota.

For authenticated requests, quota identity is the internal user ID. For guests, it is a SHA-256 hash of the normalized client IP plus a server-side quota salt; plaintext IP addresses are not persisted. The app uses a trusted-proxy configuration before consulting `X-Forwarded-For`; otherwise it uses the direct remote address. Quota rows are unique on identity type, identity hash/value, and UTC day so concurrent requests cannot bypass the limit.

## Error handling and security

- OAuth provider failures return a generic sign-in error and are logged without tokens or authorization codes.
- Missing Google OIDC configuration disables sign-in with a clear operator-facing health/configuration status; it does not weaken guest restrictions.
- Account-only APIs reject unauthenticated requests before reading an upload body or touching RAG state.
- Session cookies are `HttpOnly`, `Secure` outside local development, and `SameSite=Lax`.
- CSRF protection is applied to state-changing browser routes; OAuth callback state is validated by Authlib.

## Testing acceptance criteria

- Google OIDC callback handling is tested with mocked provider responses, including invalid state and missing subject failures.
- Two accounts cannot read, write, retrieve, clear, or receive citations for each other's drafts, logs, or RAG documents.
- Guest data is never inserted into account tables and disappears after session reset.
- Guests receive `401` for account-only APIs; signed-in users can use them.
- Exactly 20 successful AI requests are accepted per identity/day and the 21st returns `429`; identities do not share quota.
- Concurrent requests for different users query independently filtered retrievers.

## Out of scope

Subscription billing, account recovery beyond Google, document-level deletion UI, retrieval-quality evaluation, full app modularization, production WSGI deployment, and dependency/CI hardening are later integration stages.

# Public browser workspaces

Approved in conversation on 2026-09-10: remove Google OIDC and allow immediate access with isolated browser workspaces. The user requested a precise implementation plan.

## Behavior

Every visitor can open the application and use its existing chat, draft, knowledge-base, OCR, and statistics features without signing in. Existing provider configuration, validation errors, and daily request limits continue to apply. This change does not add an audio route: none is registered in the current app.py, despite references in the old authentication helper.

The existing signed Flask session carries one opaque `user_id`, generated with `secrets.token_urlsafe(32)`. All private SQL rows, vector filters, agent context, and checkpoint thread IDs use that identifier. No identity comes from a query parameter, submitted JSON, email, IP address, or old `authenticated_user_id`. IP addresses are used only for quota accounting.

History and drafts live in SQLite, not in the cookie. The cookie is a bearer capability for the workspace. Retain the current cookie lifetime and security settings; continuity lasts while that session cookie is retained. Clearing cookies or using a different browser creates another workspace, with no account recovery or cross-device synchronization. No new permanent-cookie policy is introduced.

## Removal and compatibility

Remove auth.py, its Authlib dependency, OIDC configuration/readiness reporting, authentication routes, decorators, context injection, and the Google sign-in strip. Old authentication URLs return 404. Preserve SECRET_KEY requirements and secure cookie behavior.

Keep existing Google-owned SQL rows, user tables, vectors, and checkpoints untouched. Stop creating the Google users table for new installations. Do not reassign legacy account IDs or expose their records to anonymous visitors. Ignore and remove obsolete authentication cookie keys. An existing anonymous `user_id` can continue to own its existing anonymous records. A legacy cookie draft may migrate once to that same anonymous ID only if no draft row already exists.

All visitors use the existing salted direct-peer-IP daily quota. Preserve atomic reservations, UTC rollover, configured limit (default 20), error formats, and validation-before-charge behavior. Visitors on the same IP share the allowance; a reverse proxy may cause more sharing unless separately configured. Proxy trust changes are outside this work.

## Verification and limits

Tests must prove two independent browsers cannot read, search, retrieve, delete, or clear each other's data and never share checkpoint threads. Old account cookies must not recover old account data. Clearing chat clears both display history and its checkpoint thread using existing error behavior. Provider-dependent success paths use mocks, not paid APIs or live models.

No database deletion, account-data migration, deployment changes, new retention job, provider replacement, public legal corpus redesign, or unrelated UI changes. Existing anonymous records may remain on disk after cookies are lost; document this limitation rather than implying automatic deletion.

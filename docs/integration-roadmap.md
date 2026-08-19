# Project Integration Roadmap

This tracker records the integration status and implementation order for production readiness. Status reflects `main` as of 2026-08-20.

| Priority | Integration | Status | Next deliverable |
| --- | --- | --- | --- |
| 1 | Google OIDC accounts, data ownership, and quotas | Planned | Implement the approved [design](superpowers/specs/2026-08-19-google-oidc-user-foundation-design.md). |
| 2 | Generated-report HTML sanitization | Integrated | Bleach allowlist is active for OCR and statistics reports. |
| 3 | Upload and expensive-route hardening | Not started | Add size, MIME/signature, document-complexity, and rate-limit controls. |
| 4 | Production serving and configuration | Not started | App factory, environment profiles, WSGI server, secure logging. |
| 5 | True POST-based streaming chat | Not started | Background token production, cancellation, timeouts, no prompt-bearing GET URL. |
| 6 | RAG lifecycle and answer quality | Partially integrated | Add citations, deduplication, document-level deletion, and evaluation after ownership isolation. |
| 7 | Application modularization | Not started | Split routes, services, persistence, and configuration around the new boundaries. |
| 8 | Reproducible dependencies and CI | Not started | Pin dependencies, define Python support, and add test/lint/security workflows. |
| 9 | Test expansion | Partially integrated | Add authentication, quota, upload-defense, RAG-isolation, and failure-path coverage. |
| 10 | STT parameter bug | Not started | Rename the invalid `language_code` keyword at the caller to `language`. |

## Completed foundation work

- Anonymous browser-session IDs isolate the current SQLite draft/chat rows and uploaded RAG metadata.
- Server-side report HTML sanitization uses a strict Bleach allowlist.

The anonymous isolation is not sufficient for multi-user deployment because the RAG retriever remains mutable process-global. Priority 1 replaces it with authenticated, request-scoped ownership.

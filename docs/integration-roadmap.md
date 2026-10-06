# Project Integration Roadmap

> **2026-10-06:** The LangGraph agent, its checkpointer and `evals/` described below were removed. Web chat now sends every turn to legal_rag `/v1/chat`. This page is kept as a record.

This tracker records the integration status and implementation order for production readiness. Status reflects `main` plus the LangGraph agent migration on `feat/langgraph-rag-agent`, as of 2026-09-06.

| Priority | Integration | Status | Next deliverable |
| --- | --- | --- | --- |
| 1 | Google OIDC accounts, data ownership, and quotas | Integrated | Google OIDC, account/guest boundaries, 20-request daily quotas, and request-scoped RAG retrieval are covered by tests. |
| 2 | Generated-report HTML sanitization | Integrated | Bleach allowlist is active for OCR and statistics reports. |
| 3 | Upload and expensive-route hardening | Not started | Add size, MIME/signature, document-complexity, and rate-limit controls. |
| 4 | Production serving and configuration | Not started | App factory, environment profiles, WSGI server, secure logging. |
| 5 | True POST-based streaming chat | Not started | Background token production, cancellation, timeouts, no prompt-bearing GET URL. The streaming *producer* was rewritten during the LangGraph migration; the *transport* was not. |
| 6 | RAG lifecycle and answer quality | Advanced | Citations, deduplication, and evaluation are in place. Remaining: retrieval-quality measurement on fixtures larger than one chunk, and per-source lifecycle beyond the existing delete route. |
| 7 | Application modularization | Not started | Split routes, services, persistence, and configuration around the new boundaries. |
| 8 | Reproducible dependencies and CI | Not started | Pin dependencies, define Python support, and add test/lint/security workflows. |
| 9 | Test expansion | Partially integrated | Add authentication, quota, upload-defense, RAG-isolation, and failure-path coverage. |
| 10 | STT parameter bug | Not applicable | Audio processing was removed from `main`; this bug now belongs to `feat/local-audio-document`. |
| 11 | Korean legal RAG backend (`src/legal_rag`) | V0.1–V0.3 integrated | Data foundation, hybrid retrieval with evaluation, verified RAG chat over CLI and FastAPI are on `main` with 106 offline tests. Next: V0.4 QLoRA (SFT dataset, Transformers backend) and V0.5 temporal and claim-level verification. See `superpowers/specs/2026-09-07-korean-legal-rag-backend-design.md`. |

## Completed foundation work

- Google OIDC accounts own persistent SQLite drafts/chat history and RAG chunks; guests retain only a non-persistent session draft.
- Per-account RAG retrievers and document metadata prevent one account's document chunks from being retrieved by another request.
- Server-side report HTML sanitization uses a strict Bleach allowlist.

## Priority 6 — what the LangGraph migration delivered

The chat agent moved off the deprecated `langchain_classic.initialize_agent` onto
`langchain.agents.create_agent` (LangGraph). Against Priority 6 specifically:

- **Citations.** `search_documents` prefixes every returned passage with
  `[source: <filename>]`, so the answer's evidence is nameable.
- **Deduplication.** The tool drops repeated passage bodies before returning them.
  It also retrieves *once*; the `RetrievalQA` + `load_qa_chain` pair it replaced
  retrieved twice per call and spent an extra LLM call summarizing passages the
  agent was about to read anyway.
- **Evaluation.** `evals/dataset.py` defines 16 golden cases (14 answerable, 2
  unanswerable) over 3 synthetic fixtures; `evals/run_eval.py --provider
  ollama` runs them and exits non-zero when cases fail. It replaces
  `langsmith_eval.ipynb`, which was neither diffable nor runnable in CI and has
  been deleted.
- **Observability.** LangSmith tracing is opt-in and off by default; runs carry
  `thread_id`, `llm_provider`, `llm_model`, `embedding_provider`, and `collection`.

Document-level deletion, which Priority 6 previously listed as outstanding, was
already delivered before this branch (`delete_knowledge_base_source()` and the
`DELETE /knowledge_base/source` route, in `4e688e4`).

The headline evaluation finding — a 4B model reads a *conditional* tool
instruction literally, so retrieval must be made unconditional (correctness
4/16 → 15/16, tool use 4/16 → 16/16) — is written up in the README, section 5.

## Priority 5 — explicitly still open

The LangGraph migration changed how streamed chat is *produced* and deliberately
did not change how it is *transported*:

- **Changed:** SSE events now come from `agent.stream(..., stream_mode=["messages",
  "updates", "values"])`. The previous `BaseCallbackHandler` does not fire under
  LangGraph, so the producer had to be rewritten.
- **Unchanged:** the wire contract is identical, which is why `static/lc-chat.js`
  was never touched. Chat is still served over the prompt-bearing
  `GET /chat?message=...`, and **no cancellation and no timeouts were added.**

Priority 5 therefore stays `Not started`. Converting to POST, moving token
production to a background worker, and adding cancellation and timeouts remain
entirely to do.

## Repository hygiene and outstanding credential rotation

Three paths were tracked in git despite being listed in `.gitignore`, and were
untracked during this work:

- `.env` (`92e9adc`)
- `chroma_db/` (`1bf6c9a`)
- `sqlite_db/app_state.db` (`966cb98`)

Untracking removes them from future commits. It does **not** remove them from
history, and the history was already pushed to the origin remote.

> **Outstanding user action — not fixed by any code change in this work:**
> every credential that was previously committed in `.env` and pushed to origin
> must be treated as disclosed and **rotated at its provider**. That covers every
> secret the file contained — the Google API key, the SerpAPI key, the Google
> OIDC client secret, the Flask `SECRET_KEY`, and `QUOTA_IP_SALT`. The Google
> API and SerpAPI wiring has since been removed, so those two keys are no longer
> read by anything; they were still disclosed and should still be revoked at
> their provider rather than merely abandoned. Rotating them is the only remedy;
> purging history is optional cleanup and is not a substitute.


## Google API and SerpAPI removal

The Gemini chat/embedding connector and the SerpAPI web-search tool were removed
after the migration. The app now runs entirely on local Ollama models and reads
no third-party AI credential.

- `llm_model.py` supports `ollama` and `none`. `none` is an explicit off switch,
  added because Ollama needs no credential and would otherwise leave no way to
  run with AI disabled — a state the app's degradation paths and tests rely on.
- `EMBEDDING_PROVIDER` now defaults to `ollama`, so the active collection is
  `docs_ollama_nomic`. An existing Google-embedded knowledge base is not migrated
  and reads as empty until re-ingested.
- `search_documents` is the entire tool budget; there is no web search.
- `langchain-google-genai` and `google-search-results` were dropped from
  `requirements.txt`.
- Google OIDC sign-in is unaffected — it is authentication, not an AI API.

## Next steps

The next production-readiness work is upload and expensive-route hardening
(Priority 3), followed by POST-based streaming (Priority 5). The LangGraph agent
migration design and plan are in `docs/superpowers/specs/` and
`docs/superpowers/plans/`.

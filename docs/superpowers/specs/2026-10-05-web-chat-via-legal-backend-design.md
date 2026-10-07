# Web chat answers through the legal_rag backend

Approved 2026-10-05.

## Problem

The web chat (Flask `/chat`) and the legal_rag API (`/v1/chat`) are two answer
pipelines. The web chat calls `/v1/search`, then lets the local agent
(qwen3:4b) write the reply, checked only by `citation_guard`. The backend
pipeline has the configurable generator (Gemini by default), the verifier and
the regenerate/strip/reject policy, and it is the pipeline every answer
evaluation measures. Users of the web chat therefore get neither the default
generator nor the verified answer, and the scores do not describe what they see.

## Decision

Every web chat message is answered by `/v1/chat`. The Flask agent no longer
takes part in a chat turn. Document work (uploads in chat, statement and report
suggestions) is unused and is not preserved in the chat route.

## Behaviour

- Each message is posted to `LEGAL_RAG_URL` + `/v1/chat` with a conversation id
  built from the browser session id and an epoch that clearing the chat advances.
- Follow-up turns: when the conversation has earlier turns, the backend rewrites
  the question into one that stands alone (`rag/condenser.py`, one extra model
  call, only after the clarification check) and retrieves, reranks and answers
  with that question; generation also gets a short history block. A new
  conversation makes no extra call. Before this, retrieval saw only the
  current message and a follow-up such as "그 신청은 어디에 하나요?" retrieved
  unrelated chunks.
- Open: the analyzer can ask for more facts again after the user answered, so a
  conversation can clarify several times in a row (8 of 15 two-turn probes).
  There is no limit on clarification rounds yet.
- The generator is whatever `LEGAL_RAG__LLM__BACKEND` / `LEGAL_RAG__LLM__MODEL`
  select. The web chat does not read `LLM_PROVIDER`.
- Short greetings and thanks get a fixed reply and do not call the backend.
- `kind: answer` is rendered as summary, analysis, possible actions, further
  facts needed and uncertainty. Evidence items fill the existing sources panel.
- `kind: clarification` shows the backend's questions.
- `kind: rejected` tells the user the answer could not be grounded.
- The SSE event types stay the same (`thought`, `final_response`, `error`,
  `stream_end`), so the front end is unchanged. There is no token streaming;
  the reply arrives whole. The three suggestion fields are sent empty.
- `citation_guard` still annotates the final text as a second check.
- The daily request quota and chat history storage are unchanged.

## Failure

If `LEGAL_RAG_URL` is unset, or the backend is unreachable, times out or
returns an error status, the user sees a message that the legal answer service
is unavailable. There is no fallback to the local agent.

## Out of scope

Removing the agent, upload and drafting code. It stays in the repository,
unreached by the chat route, and is a separate cleanup. The exception is the
agent stream generator in `app.py` and the tests that drove it through `/chat`:
once unrouted they could not be exercised, so they were removed.

## Tests

Unit tests with a stubbed backend for the four outcomes (answer, clarification,
rejected, failure) and for the renderer. Existing tests that assert the forced
pre-search or agent streaming in the chat route are updated to the new
behaviour.

# Answer streaming: show the analysis as a draft

Date: 2026-10-17. Status: approved by the user.

## Problem

A web chat turn shows nothing until the whole answer has been written and verified:
17.5 s median, 30.7 s p90 (`data/eval/reports/chat_pipeline/latency-e2e-2026-10-10/`).
About 60% of a turn is the model writing the answer. Cutting output failed twice
(see `docs/experiments`), so only the perceived wait can move.

## Decision

Stream the `analysis` field to the browser while the model writes it. Show it as a
draft, labelled "작성 중 · 검증 전". When verification finishes, replace the draft
with the final answer, built exactly as today. The user chose this over showing only
progress steps and over showing unlabelled text.

## Probe (2026-10-17, Ollama 0.40.4, qwen3:4b, flash attention on)

- `stream: true` with a JSON-schema `format` works. Content arrives as JSON
  fragments in field order, starting with `{"analysis": "...`.
- `message.thinking` stayed empty and no reasoning reached `content`.
- The `done: true` chunk carries `eval_count`, `prompt_eval_count` and the durations.

## Flow

1. `OllamaChatModel.generate(..., on_text=None)`. With a callback it requests
   `stream: true`, passes each content fragment to the callback, and joins them.
   Parsing and the counter log line do not change. Without a callback the request is
   the same as today. Other model backends accept the argument and ignore it.
2. `Generator.generate(..., on_draft=None)`. Feeds the fragments to a small decoder
   that emits the decoded value of `"analysis"` as it grows. The decoder holds back an
   escape split across fragments (`\`, part of `\uXXXX`). The repair retry is a second
   model call, so it starts a new draft.
3. `Orchestrator.handle(..., on_draft=None)`. Sends a reset before each generation
   attempt: a regeneration, a repair retry, or a failure that falls back to an earlier
   draft. The analyzer, condenser and rewriter never stream. With no callback,
   `handle()` behaves as today, so `frozen.py`, `judge_answers.py`,
   `eval_chat_pipeline.py` and `scripts/chat.py` are not affected.
4. New endpoint `POST /v1/chat/stream`, NDJSON. It runs `handle` in a worker thread,
   which feeds a queue. Lines: `{"type": "draft", "text": delta}`,
   `{"type": "draft_reset"}`, and finally `{"type": "response", "body": ...}`. The
   body is the same ChatResponse JSON as `/v1/chat`, with `exclude_none` and the trace
   removed. `/v1/chat` stays as it is.
5. Flask: `legal_backend` reads the stream with `httpx.stream(...).iter_lines()`.
   The timeout now limits each read, not the whole turn. The SSE route forwards
   `draft` and `draft_reset`. The final body goes through `render`, the second
   citation check, the annotations and the DB write, as today.
6. Browser (`lc-chat.js`): keeps the draft text, renders it dimmed with the label, and
   removes `S1`/`D1` source markers from it. `final_response` replaces the draft.
   An error or a rejected answer removes it. A clarification never starts one.

## Display rules

- The draft has no source numbers. The final numbers ([1], [2]) depend on which
  sources the answer cites, which is known only at the end.
- The draft shows only the analysis. The final answer puts the summary first, so
  the order changes when it is replaced. This is intended.
- Time to the first draft character is estimated at 6–11 s (analyzer, retrieval,
  prompt reading). This is a target to measure, not a promise.

## Tests

- Unit: the decoder, fed a real answer split at every byte offset, emits exactly
  the decoded analysis. The endpoint emits drafts, then resets, then one response.
  Flask forwards draft events and handles the final body as before.
- Pre-registered latency run with the `latency-e2e-2026-10-10` driver. `run.py`
  also records the time of the first draft event. Check the Ollama log for
  `OLLAMA_FLASH_ATTENTION:true` before the run.
  - (a) Time to the first draft character: median and p90.
  - (b) Time to the final answer must not be worse than `baseline.json`.
- Answers do not change: the same questions with a fixed seed, streaming on and
  off, compared as text. If a seed does not make output repeatable on this GPU, check
  instead that there are no new generation failures and that tokens/s is unchanged.
  Do not regrade: the grader flips grades between identical runs.

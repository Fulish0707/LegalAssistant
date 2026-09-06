# LangGraph RAG Agent — Structural Design

Date: 2026-09-06
Status: Proposed

## Purpose

Move the chat agent off the deprecated `langchain_classic.initialize_agent` onto
`langchain.agents.create_agent` (LangGraph), restore conversation memory, and put
the model provider behind a connector so Qwen3-4B running locally can serve as the
primary model with Gemini as a comparison baseline.

Scope is the chat and tool-loop path: `langchain_config.py` and its callers in
`app.py`. `llm_ocr.py` and `statistics_analysis.py` are one-shot generation, not
agent loops, and are excluded.

This design builds on branch `google-oidc-user-foundation`, which already replaced
the process-global retriever with `build_agent(user_id)` and `build_rag_tool(user_id)`
factories. Those boundaries are the foundation, not the problem.

## 1. Model connectors

`llm_model.py` collapses to two env-driven factories:

```python
get_chat_model()       # LLM_PROVIDER=ollama | google
get_embedding_model()  # EMBEDDING_PROVIDER=ollama | google
```

Nothing outside these two functions names a provider.

### Runtime: Ollama, not in-process transformers

Qwen3-4B is served through Ollama and reached via `ChatOllama`. Measured
constraints on this machine drive that choice:

- **8 GB VRAM (RTX 4060 Laptop).** Qwen3-4B at bf16 is roughly 8 GB of weights
  alone and will not fit alongside a KV cache. A Q4_K_M GGUF is roughly 2.5 GB and
  leaves ample room for context.
- **Already installed.** Ollama 0.16.2 and `langchain-ollama==1.1.0` are present.
  vLLM is not, and is a poor fit for a single 8 GB laptop GPU.
- **Tool calls are actually parsed.** `ChatOllama` implements `bind_tools` and
  surfaces `tool_calls` as structured fields. In-process `HuggingFacePipeline`
  wrapped in `ChatHuggingFace` does not reliably parse Qwen3's `<tool_call>` blocks
  back into tool calls, which is what `create_agent` requires.

`langchain-huggingface` is therefore not a dependency. Direct `transformers`
loading stays available as a later provider branch if the runtime changes.

### Qwen3-specific settings, owned by the connector

- **Leave `reasoning` unset.** All three settings were measured on `qwen3:4b`,
  and each trades away something:

  | setting | `content` | thinking | `structured_response` |
  | --- | --- | --- | --- |
  | unset (default) | clean | dropped | populated 2/3 (simple schema) |
  | `reasoning=True` | clean | isolated in `reasoning_content` | **0/3 — suppressed** |
  | `reasoning=False` | **thinking leaks in as prose** | — | not measured |

  `reasoning=False` reads like the safe choice and is the only one that pollutes
  output: it strips the `<think>` tags and leaves reasoning text in `content`.
  `reasoning=True` looks better still — thinking stays inspectable in traces — but
  it suppressed the structured-output tool call in every run. Unset is the only
  setting where that path works at all. `OLLAMA_REASONING=true` trades it back for
  trace visibility.

- **Do not use `temperature=0`.** The current code hardcodes it, and greedy
  decoding is a known degradation and repetition mode for Qwen3. Sampling
  parameters are per-provider and belong in the connector, not at the agent call
  site.
- **Context.** Qwen3-4B is 32K native, comfortable for `k=3` retrieval over
  1000-character chunks.

### Embedding dimension — a demonstrated hazard, not a theoretical one

The existing collection is `langchain` at 768 dimensions, written by
`text-embedding-004`. The already-pulled local embedder `nomic-embed-text` was
measured at **768 dimensions as well**.

Chroma will therefore accept nomic vectors into the existing collection without
error, and retrieval will silently compare vectors from two unrelated embedding
spaces. Wrong results, no exception, no log line.

The collection name is therefore derived from provider and model —
`docs_google_te004`, `docs_ollama_nomic` — so switching connectors creates a new
collection. Re-indexing is explicit and never implicit. This single rule is what
makes the embedding connector safe.

### Capability gate

`create_agent` requires a model implementing `bind_tools`. Verified on this
machine: `qwen3:4b` through `ChatOllama` returns populated, structured `tool_calls`
rather than raw `<tool_call>` prose. The gate is therefore insurance rather than the
main path, but it stays for the provider branch that is not yet chosen:

```python
model = get_chat_model()
if not hasattr(model, "bind_tools"):
    # fixed retrieve -> generate chain, no tool loop
```

## 2. Agent

`create_agent(model, tools, system_prompt=..., checkpointer=..., context_schema=...)`
replaces `initialize_agent`.

- `user_id` moves from a closure capture into LangGraph's typed `context_schema`.
- The RAG tool is registered only for signed-in accounts, as it is today.
- **Structured output replaces marker parsing.** `app.py` currently asks the model
  for freeform `[CHAT]` / `[STATEMENT_SUGGESTION]` / `[JOB_STATEMENT_SUGGESTION]` /
  `[REPORT_SUGGESTION]` markers and recovers them with a regex. A 4B model follows
  that convention far less reliably than Gemini. Declaring the same four fields as
  `response_format` on `create_agent` makes the contract schema-enforced.

  **But a 4B does not honour it reliably, so the regex parser cannot simply go.**
  Measured on `qwen3:4b`: with a one-field schema, `structured_response` populated
  in 2 of 3 runs; with the real four-field `AgentSections` plus the full system
  prompt, it populated in **0 of 3**. The tool loop itself never failed — the model
  called `search_documents` every time — and the final message content was valid
  schema JSON every time. What fails is routing that JSON through the
  structured-output tool call.

  `ProviderStrategy` was tried and rejected: it populates `structured_response`
  reliably, but Ollama's `format=json_schema` constrains every generation to the
  schema, so the model can never emit a tool call and the RAG tool becomes
  unreachable. That defeats the migration.

  The resolution is `extract_agent_sections()`: read `structured_response` first
  (Gemini's route), then fall back to parsing JSON out of the final message content
  (Qwen's route), then fall back to the legacy markers. Parse before any cleanup
  pass that unwraps code fences, which would corrupt valid JSON. The fallback is
  load-bearing, not belt-and-braces — it is what carries every Qwen request.
- **Keep the tool set small.** A 4B model degrades quickly as tool count grows.
  `search_documents` plus optional web search is the budget.

## 3. Memory

The OIDC branch deleted `ConversationBufferMemory` and now passes `chat_history=[]`,
so the agent is stateless between turns. A LangGraph `SqliteSaver` checkpointer over
the existing `sqlite_db/` restores memory, keyed by `thread_id = user_id`.

SQLite `chat_history` stays the display record. The checkpointer is the agent's
working state. The two are separate on purpose.

## 4. RAG tool

Renamed `RAG Retriever` -> `search_documents`. The space breaks function-calling
schema validation (`^[a-zA-Z0-9_-]+$`), and Qwen3's tool template is stricter about
tool names than Gemini, not looser.

The body becomes deterministic: retrieve, deduplicate, format with citations. This
deletes `RetrievalQA` and `load_qa_chain`, and removes the double retrieval in the
current `rag_tool_func`, which retrieves once directly and then again inside the
chain. Determinism matters more with a 4B model: every step moved out of the model
and into code is a step it cannot get wrong.

Output is structured (text plus source metadata) so a relevance grader can later be
added as middleware without reshaping the graph, and so evaluators have a stable
target.

## 5. LangSmith

`langgraph==1.2.11` and `langsmith==0.11.0` are already installed transitively and
need pinning in `requirements.txt`, alongside `langchain-ollama==1.1.0`.

- **Tracing:** env-driven (`LANGSMITH_TRACING`, `LANGSMITH_API_KEY`,
  `LANGSMITH_PROJECT`). Runs carry `thread_id` so turns group into conversations,
  and carry the model provider and model name.
- **Datasets and evaluation:** a golden question/answer set over fixture documents,
  with correctness and groundedness evaluators, runnable from a script rather than a
  notebook. `langsmith_eval.ipynb` uses `langchain_teddynote` and raw
  `google.generativeai`; it is superseded, not extended.

Evaluation is what makes the local move decidable. Qwen3-4B against Gemini on one
dataset answers whether the 4B model holds quality on this task, and where it fails
first — most likely tool-call formatting and multi-hop retrieval. Without it, the
swap is a guess.

## 6. Streaming

Folded into this design rather than deferred: the migration breaks chat without it.
`_generate_agent_response_stream` produces SSE events from a queue fed by
`BaseCallbackHandler` (`on_agent_action`, `on_tool_end`, `on_agent_finish`). Those
callbacks do not fire under LangGraph, so the producer must be replaced.

The producer becomes `agent.stream(..., stream_mode=["messages", "updates"])`:
`messages` chunks become `{"type": "token"}` events, and `updates` node output
becomes the `{"type": "thought"}` events the UI already shows for tool activity.

**The wire contract does not change.** `static/lc-chat.js` consumes
`{type: token|thought|error|stream_end}` over `EventSource`, and it keeps consuming
exactly that. Only the producer is rewritten, so the frontend is untouched.

Deliberately out of scope: converting the prompt-bearing `GET /chat?message=...`
into true POST streaming, with cancellation and timeouts. That remains roadmap item
5. This design keeps the existing POST-acknowledge-then-GET-stream flow and swaps
only what LangGraph forces it to swap.

## 7. Import-time side effects

`langchain_config.py` currently builds the vector store, loads documents, and
constructs the agent at import. That is why the globals existed. Ingestion moves
behind an explicit call so importing the module performs no I/O and needs no API
key — which also unblocks `tests/test_app_contract.py`.

## Verified on this machine

- `qwen3:4b` via `ChatOllama` returns structured `tool_calls`.
- `reasoning=True` keeps `content` clean; `reasoning=False` does not.
- `create_agent` tool loop and `response_format` work together.
- `nomic-embed-text` is 768-dim, colliding with the existing collection.

## Risks

- **Streaming UX tension.** With `response_format`, the final answer arrives as a
  structured tool call rather than streamed prose, so the four sections cannot
  themselves be token-streamed. Accepted: stream tokens and tool activity for
  progress, then emit the structured sections as one final event. The alternative —
  dropping `response_format` and going back to regex markers — trades a schema
  guarantee for streaming polish, and the 4B is exactly the model that makes that
  a bad trade.
- **Latency.** A local 4B on a laptop GPU is slower per token than the Gemini API,
  and an agent loop multiplies that across turns. Worth measuring before committing
  the user-facing path.
- **Merge hazard.** The OIDC branch's `requirements.txt` is unpinned and stale
  against main's pinning work. Resolve toward main's pins.
- **Python version drift.** `requirements.txt` targets Python 3.10; the active
  interpreter is 3.13.9. Reconcile before pinning anything further.

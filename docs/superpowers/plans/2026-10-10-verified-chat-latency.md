# Verified Chat Latency Implementation Plan

> **For agentic workers:** Execute inline with `superpowers:executing-plans`. Preserve existing edits. Do not start subagents, commit, install dependencies, or run tests without an applicable user request.

**Goal:** Reduce work before the verified answer is displayed, preserving the current validation path.

**Architecture:** Reuse identical query embeddings within each hybrid retrieval call. Add a configurable concise expression instruction to the existing complete answer schema. Record generation timings without logging content.

**Tech Stack:** Python, Pydantic, Ollama/httpx, Qdrant, Flask/FastAPI.

**Design:** [Verified-only design](../specs/2026-10-10-chat-latency-design.md).

## Task 1. Reuse query embeddings

**Files:** `src/legal_rag/retrieval/dense.py`, `src/legal_rag/retrieval/hybrid.py`.

- [x] Make `DenseRetriever.retrieve` delegate to `retrieve_vector`:

```python
def retrieve(self, query: str, k: int, filters: RetrievalFilters | None = None) -> list[ScoredChunk]:
    return self.retrieve_vector(self.embedder.embed_query(query), k, filters)

def retrieve_vector(self, vector: list[float], k: int, filters: RetrievalFilters | None = None) -> list[ScoredChunk]:
    return [ScoredChunk(chunk=chunk, score=score, retriever=self.name)
            for chunk, score in self.store.search_dense(vector, k, filters)]
```

- [x] Import `DenseRetriever` in `hybrid.py`. At the start of `retrieve_many`, create `query_vectors: dict[tuple[int, str], list[float]] = {}`.
- [x] Replace only the per-leg retrieval call with:

```python
if type(leg) is DenseRetriever:
    key = (id(leg.embedder), query)
    if key not in query_vectors:
        query_vectors[key] = leg.embedder.embed_query(query)
    hits = leg.retrieve_vector(query_vectors[key], self.k_leg, filters)
else:
    hits = leg.retrieve(query, self.k_leg, filters)
```

Concrete-type matching preserves custom retrievers and subclasses with their own `retrieve` behavior. Keep ranked-list order, trace entries, fusion and reranking intact. No persistent cache.

## Task 2. Add concise expression without shrinking the answer contract

**Files:** `src/legal_rag/config/settings.py`, `src/legal_rag/rag/generator.py`, `src/legal_rag/rag/orchestrator.py`, `src/legal_rag/llm/prompts/answer_concise.md`, `configs/local.yaml`.

- [x] Add `answer_style: Literal["full", "concise"] = "full"` to `RagSettings`.
- [x] Add keyword-only `answer_style: str = "full"` to `Generator.__init__`; reject other values with `ValueError` and assign `self.answer_style`.
- [x] In `_generate_structured`, build the system text before constructing the messages:

```python
system = load_prompt("answer_system")
if self.answer_style == "concise":
    system += "\n\n" + load_prompt("answer_concise")
```

Use `system` for the existing system message. Leave the remaining messages and retry loop intact. Plain-text adapter generation retains its existing prompt.

- [x] Pass `answer_style=settings.rag.answer_style` when the orchestrator builds `Generator`.
- [x] Write `answer_concise.md` with this content:

```text
간결한 표현 지침:
- 위의 근거 사용 규칙과 답변 규칙을 모두 지키세요. 짧게 쓰기 위해 결론을 바꾸는 조건, 예외, 적용 범위, 기간 또는 필요한 불확실성을 생략하지 마세요.
- summary는 핵심 결론과 중요한 전제를 한두 문장으로 쓰세요. analysis는 보통 3~5문장으로 핵심 근거와 적용 조건을 설명하되, 여러 쟁점이나 예외를 정확히 설명하려면 더 길게 쓰세요.
- summary, analysis, uncertainty에 같은 설명을 반복하지 마세요. 이미 제공된 사실이나 근거 전문을 길게 되풀이하지 마세요.
- legal_issues, possible_actions, additional_information_needed는 중복을 합치고 꼭 필요한 항목부터 쓰세요. 보통 각각 두 항목 이내로 쓰되, 결론이나 필요한 대응이 달라지는 항목은 추가하세요.
- citations의 source_id, kind, reference, quote와 confidence를 유지하세요. quote는 해당 근거에서 실제로 확인되는 짧은 구절을 그대로 인용하세요. 조건이나 예외를 잘라 의미를 바꾸지 마세요.
- 확인하지 못한 내용은 uncertainty에 구체적으로 밝히세요. 필요한 제한 설명은 남기고 같은 일반적 주의 문구를 반복하지 마세요.
- 짧은 답변이 정확한 답변과 충돌하면 정확한 답변을 우선하세요. 모든 필드를 기존 JSON 스키마에 맞게 작성하세요.
```

- [x] Initially add `rag: {answer_style: concise}` to the local profile. The measurement record below restores `full`. Do not lower `max_tokens`, remove citations, change field order, or disable regeneration.

## Task 3. Record content-free inference timings

**Files:** `src/legal_rag/llm/ollama.py`, `scripts/serve.py`.

- [x] Import `logging` and `time`, and define `logger = logging.getLogger(__name__)`.
- [x] Capture `started = time.perf_counter()` immediately before the HTTP call.
- [x] After decoding the response body, emit one INFO record:

```python
logger.info(
    "ollama_generation model=%s schema=%s wall_ms=%.1f total_ms=%.1f "
    "load_ms=%.1f prompt_ms=%.1f generation_ms=%.1f "
    "prompt_tokens=%s output_tokens=%s stop=%s",
    self.model, response_schema.__name__ if response_schema else "text",
    (time.perf_counter() - started) * 1000,
    (body.get("total_duration") or 0) / 1_000_000,
    (body.get("load_duration") or 0) / 1_000_000,
    (body.get("prompt_eval_duration") or 0) / 1_000_000,
    (body.get("eval_duration") or 0) / 1_000_000,
    body.get("prompt_eval_count"), body.get("eval_count"), body.get("done_reason"),
)
```

Do not log messages, evidence, the generated text, API keys, or raw response bodies. Retain `stream: false` and the existing parsing contract.

- [x] Configure the timing namespace in `scripts/serve.py` inside `main`, after settings load and before `uvicorn.run`. Add `import logging` at module scope:

```python
timing_logger = logging.getLogger("legal_rag.llm.ollama")
if not timing_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    timing_logger.addHandler(handler)
timing_logger.setLevel(logging.INFO)
timing_logger.propagate = False
```

This exposes timing counters during the standard API launch without enabling all library INFO logs. Alternative API launchers must configure this namespace themselves.

## Task 4. Review and integrate the patch

- [x] Develop in `.worktrees/verified-chat-latency`, branch `perf/verified-chat-latency`.
- [x] Review the isolated diff and parse modified Python files with `ast.parse`; this is a static source check, not an inference or application test.
- [x] Check patch whitespace with `git diff --check`.
- [x] Apply only the reviewed implementation patch to the shared checkout after confirming each touched file still matches its captured HEAD baseline.
- [x] Confirm existing changes in `llm_ocr.py`, `tests/test_document_citation_notice.py`, and `docs/superpowers/plans/2026-10-09-four-levers.md` retain their original hashes.
- [x] Report implemented changes separately from measured outcomes. The implementation-stage request did not ask for tests, so none were added or run at that stage. The later measurement request authorizes the focused tests below. Do not claim speed or legal-quality improvement without evidence.

## Subsequent measurement

Use the design's acceptance criteria when performance and quality verification is requested. Compare embedding-only optimization and concise expression separately. Keep model, evidence, context size, output cap, and verifier constant; use repeated runs, report failures and condition omissions. Any GPU run must coordinate through the existing GPU lock. The API requires a restart to read the changed local profile; do not restart the user's running services automatically.

## Initial implementation record

2026-10-10: Tasks 1–4 implemented and the reviewed patch applied to the shared checkout. Python AST parsing and `git diff --check` completed without errors. The shared implementation matches the isolated worktree. Existing user edits and the display, answer schema, verification, token-cap and regeneration contracts were preserved. No application tests, model inference, latency benchmark, or answer-quality evaluation was run. No service was restarted and no commit was made.

At that point the local profile selected concise expression. The later measurement decision below supersedes that default. Query-vector reuse remains active independently of the expression setting.

## Measurement record

User request: “move onto measuring that”. This authorizes benchmarks and focused regression tests.

- [x] Write measurement criteria before model calls in `data/eval/reports/chat_pipeline/verified-latency-2026-10-10/PREREG.md`.
- [x] Compare 48 retrieval pairs with the original method from `d03fd46`. All leg, fused and final orders match; score tolerance is `1e-5`.
- [x] Generate complete answers for all 80 development questions with each style and fixed evidence. Alternate the arm order per question.
- [x] AI review the 20 questions at every fourth index; preserve reasons and scope.
- [x] Restore `configs/local.yaml` to `answer_style: full` after finding a scope error. Record this in `AMENDMENTS.md`. Explicit style selection keeps benchmark arms unchanged.
- [x] Run 52 focused backend and 33 web tests. Confirm the cache regression fails against the original method and passes with reuse.
- [x] Finish the 20-question repeated timing comparison. Full median 15.89s, concise 15.39s; neither primary nor repeat supports the 15% improvement target. The repeat order deviation is recorded in `AMENDMENTS.md`.
- [x] Finish both reference-conclusion grading rubrics and review changed cases. Core 50/80 vs 49/80; strict 38/80 vs 41/80. All 14 changed-verdict questions reviewed; 27 distinct questions reviewed including the original 20.
- [x] Finish scratch Flask/API timing for warm, follow-up, document, unloaded-model and concurrent requests. 80 measured turns plus 5 warmups; all final responses recorded. Keep the 2 upload analyses separate.
- [x] Write a self-contained [measurement report](../../experiments/verified-chat-latency-2026-10-10.md), with limitations and the adoption decision. Warm baseline/reuse/concise medians: 21.44s / 18.66s / 23.69s. Do not attribute the entire web reduction to embedding reuse.

The concise prompt remains an experimental opt-in. It did not meet the primary timing or condition-preservation criteria. No user service restart, commit, corpus change, or index rebuild is part of this measurement.

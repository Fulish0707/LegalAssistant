# Answer Streaming Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Show the answer's `analysis` field in the web chat as a labelled draft while the model writes it, then replace it with the verified answer.

**Architecture:** One optional callback goes down the call chain: Ollama streaming → `Generator` decodes `"analysis"` from partial JSON → `Orchestrator.handle(on_draft=...)`. A new `/v1/chat/stream` endpoint sends it as NDJSON, Flask forwards it as SSE, and `lc-chat.js` renders it. `/v1/chat` and every caller without a callback are unchanged.

**Tech Stack:** FastAPI `StreamingResponse`, httpx streaming, Flask SSE, plain JS.

Design: `docs/plans/2026-10-17-answer-streaming-design.md`.

**Test commands**
- Backend: `uv run pytest tests/legal_rag -m 'not live' -q` (652 pass on main).
- Flask: `T=$(mktemp -d -p <scratchpad>); env APP_STATE_DB=$T/a.db RAG_PERSIST_DIR=$T/c APP_ENV=testing LLM_PROVIDER=none EMBEDDING_PROVIDER=none LANGSMITH_TRACING=false /home/layke/anaconda3/bin/python -m pytest webapp/tests --ignore=webapp/tests/test_ollama_connector_live.py -k 'not Live' -q` (179 pass on main). Shortened to `FLASK_TESTS` below.

**Callback contract (all layers):** `on_draft(text: str | None)`. A string is the next piece of the analysis. `None` means "a new model call starts; throw the draft away".

---

### Task 1: Ollama streams content to a callback

**Files:**
- Modify: `src/legal_rag/llm/ollama.py`
- Modify: `src/legal_rag/llm/base.py` (protocol), `fake.py`, `gemini.py`, `transformers.py` (accept `on_text=None`)
- Test: `tests/legal_rag/unit/test_llm.py`

**Step 1: Write the failing tests** (append to `test_llm.py`)

```python
def test_ollama_streams_content_to_the_callback_and_parses_the_whole():
    seen, pieces = {}, []
    lines = [{"message": {"content": '{"answer": '}}, {"message": {"content": '"ok"}'}},
             {"message": {"content": ""}, "done": True, "eval_count": 2}]

    def handler(request: httpx.Request):
        seen.update(json.loads(request.read()))
        return httpx.Response(200, content="".join(json.dumps(line) + "\n" for line in lines))

    m = OllamaChatModel(model="m", base_url="http://t", transport=httpx.MockTransport(handler))
    r = m.generate([Message(role="user", content="q")], response_schema=Out, on_text=pieces.append)
    assert seen["stream"] is True and pieces == ['{"answer": ', '"ok"}']
    assert r.parsed.answer == "ok" and r.raw["eval_count"] == 2


def test_ollama_stream_error_line_raises():
    def handler(request: httpx.Request):
        return httpx.Response(200, content=json.dumps({"error": "model crashed"}) + "\n")

    m = OllamaChatModel(model="m", base_url="http://t", transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="model crashed"):
        m.generate([Message(role="user", content="q")], response_schema=Out, on_text=lambda piece: None)


def test_fake_model_passes_its_text_to_the_callback():
    pieces = []
    FakeChatModel({"Out": ['{"answer": "x"}']}).generate([Message(role="user", content="q")], response_schema=Out, on_text=pieces.append)
    assert pieces == ['{"answer": "x"}']
```

**Step 2: Run, expect FAIL** (`TypeError: unexpected keyword argument 'on_text'`)

Run: `uv run pytest tests/legal_rag/unit/test_llm.py -q`

**Step 3: Implement**

`base.py`, in the `ChatModel.generate` protocol, after `max_tokens: int = 1024,` add:
```python
        on_text: Callable[[str], None] | None = None,
```
and `from collections.abc import Callable`. Add one line to the protocol docstring area (module docstring): "With `on_text`, a backend that can stream passes each piece of output text to it."

`ollama.py`, replace the body from `payload` to `body = response.json()`:
```python
    def generate(self, messages, *, response_schema: type[BaseModel] | None = None, temperature: float = 0.2, max_tokens: int = 1024,
                 on_text=None) -> ModelResponse:
        payload: dict = {
            "model": self.model,
            "messages": [message.model_dump() for message in messages],
            "stream": on_text is not None,
            ...unchanged...
        }
        ...unchanged...
        started = time.perf_counter()
        if on_text is None:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            body = response.json()
        else:
            body, pieces = {}, []
            with self._client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    body = json.loads(line)
                    if "error" in body:
                        raise RuntimeError(f"ollama: {body['error']}")
                    piece = (body.get("message") or {}).get("content", "")
                    if piece:
                        pieces.append(piece)
                        on_text(piece)
            # The last line carries the counters the log reads; the text is all the pieces.
            body["message"] = {"content": "".join(pieces)}
```
The log line and the parsing below stay as they are.

`fake.py`: add `on_text=None` to `generate`. Right after `scripted = queue.pop(0)`:
```python
        if on_text is not None:
            on_text(scripted.model_dump_json() if isinstance(scripted, BaseModel) else str(scripted))
```
`gemini.py` and `transformers.py`: add `on_text=None` to the `generate` signature only. They do not stream, so the draft stays empty and the final answer comes as today.

**Step 4: Run, expect PASS**, including the old `test_ollama_model_sends_schema_and_parses` (still `stream is False`).

**Step 5: Commit** `feat: Ollama passes streamed output text to a callback`

---

### Task 2: Decode the analysis while the JSON arrives

**Files:**
- Modify: `src/legal_rag/rag/generator.py`
- Test: `tests/legal_rag/unit/test_generator.py`

**Step 1: Write the failing tests** (append)

```python
import json as _json
from legal_rag.rag.generator import AnalysisDraft

_TRICKY = "첫 줄\n\"인용\" \\ 끝 😀 가"


def _draft_of(fragments):
    pieces = []
    decoder = AnalysisDraft(pieces.append)
    for fragment in fragments:
        decoder.feed(fragment)
    return "".join(pieces)


@pytest.mark.parametrize("ascii_only", [True, False])
def test_analysis_draft_is_exact_wherever_the_json_is_split(ascii_only):
    text = _json.dumps({"legal_issues": ["\"analysis\": 함정"], "analysis": _TRICKY, "summary": "요약"}, ensure_ascii=ascii_only)
    for cut in range(len(text) + 1):
        assert _draft_of([text[:cut], text[cut:]]) == _TRICKY, cut
    assert _draft_of(list(text)) == _TRICKY


def test_analysis_draft_ignores_text_after_the_analysis_ends():
    assert _draft_of(['{"analysis": "가"', ', "summary": "나"}']) == "가"


def test_generator_streams_the_draft_and_restarts_it_on_the_repair_retry():
    events = []
    good = _json.dumps({"analysis": "분석", "summary": "요약", "confidence": "medium"}, ensure_ascii=False)
    model = FakeChatModel({"LegalAnswer": ['{"analysis": "잘린', good]})
    answer = Generator(model).generate("질문", EvidenceBundle(items=[]), on_draft=events.append)
    assert answer.analysis == "분석"
    assert events == [None, "잘린", None, "분석"]
```
(Reuse the imports already at the top of `test_generator.py`; add `FakeChatModel`, `EvidenceBundle` and `pytest` if they are missing.)

**Step 2: Run, expect FAIL** (`ImportError: AnalysisDraft`)

Run: `uv run pytest tests/legal_rag/unit/test_generator.py -q`

**Step 3: Implement** in `generator.py` (`import json`, `import re` at the top):

```python
_ANALYSIS_KEY = re.compile(r'"analysis"\s*:\s*"')
_ESCAPES = {"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}


class AnalysisDraft:
    """Decodes the "analysis" string of a LegalAnswer while its JSON is still arriving.

    The model writes analysis first (LegalAnswer field order), so this is the first prose
    the web chat can show. An escape cut between two pieces waits for the next piece.
    """

    def __init__(self, emit) -> None:
        self.emit, self.buffer, self.at, self.done = emit, "", None, False

    def feed(self, piece: str) -> None:
        self.buffer += piece
        if self.at is None:
            # Inside a JSON string a quote is escaped, so this matches only the key itself.
            match = _ANALYSIS_KEY.search(self.buffer)
            if match is None:
                return
            self.at = match.end()
        text, at, buffer = [], self.at, self.buffer
        while at < len(buffer) and not self.done:
            char = buffer[at]
            if char == '"':
                self.done = True
            elif char != "\\":
                text.append(char)
                at += 1
            elif at + 1 >= len(buffer):
                break
            elif buffer[at + 1] != "u":
                text.append(_ESCAPES.get(buffer[at + 1], buffer[at + 1]))
                at += 2
            else:
                # \uXXXX, or a surrogate pair 😀 for a character outside the BMP.
                size = 12 if buffer[at + 2:at + 3].lower() == "d" and buffer[at + 3:at + 4].lower() in "89ab" else 6
                if at + size > len(buffer):
                    break
                text.append(json.loads(f'"{buffer[at:at + size]}"'))
                at += size
        self.at = at
        if text:
            self.emit("".join(text))
```
Note: `buffer[at+3:at+4]` can be empty when the piece ends there; `"" in "89ab"` is True, so `size` becomes 12, and the loop waits for more text. That is the safe direction: it waits, it does not split a pair.

`Generator.generate` and `_generate_structured`: add `on_draft=None`, and pass it through from `generate` to `_generate_structured` (plain-text mode ignores it). In the retry loop:
```python
        for _ in range(2):
            stream = None
            if on_draft is not None:
                on_draft(None)  # a new model call: what the last one wrote is thrown away
                stream = AnalysisDraft(on_draft).feed
            try:
                response = self.model.generate(messages, response_schema=LegalAnswer, temperature=self.temperature,
                                               max_tokens=self.max_tokens, on_text=stream)
```

**Step 4: Run, expect PASS**, then the whole backend suite.

**Step 5: Commit** `feat: decode the answer's analysis while it streams`

---

### Task 3: The orchestrator passes the callback down

**Files:**
- Modify: `src/legal_rag/rag/orchestrator.py`
- Test: `tests/legal_rag/integration/test_chat_turn.py`

**Step 1: Write the failing test.** Read `test_chat_turn.py` first and copy how an existing test builds the orchestrator with a `FakeChatModel` scripted for `QueryAnalysis` and `LegalAnswer`, plus the `_answer` helper. Add:

```python
def test_the_draft_streams_the_analysis_and_the_answer_is_unchanged(indexed):
    events = []
    orchestrator = <as in the neighbouring answer test>
    response = orchestrator.handle("<the neighbouring test's question>", conversation_id="c", on_draft=events.append)
    assert response.kind == "answer"
    assert events[0] is None and "".join(e for e in events if e) == response.answer.analysis
```
Also add a clarification case if the file has one: `on_draft` is never called (`events == []`).

**Step 2: Run, expect FAIL** (`TypeError: on_draft`).

**Step 3: Implement.** Add `on_draft=None` to `handle` and to `_generate_verified`. Pass it in the `_generate_verified(...)` call and to `self.generator.generate(..., on_draft=on_draft)`. One docstring line on `handle`: "`on_draft` receives the analysis as the model writes it, and None before each new attempt (see Generator)." Nothing else changes: each attempt calls the generator, and the generator sends the reset.

**Step 4: Run, expect PASS**, then the backend suite.

**Step 5: Commit** `feat: the chat turn can stream its answer draft`

---

### Task 4: `/v1/chat/stream` endpoint

**Files:**
- Modify: `src/legal_rag/api/routes.py`
- Test: `tests/legal_rag/integration/test_api.py`

**Step 1: Write the failing test.** Model it on `test_chat_endpoint_round_trip` (line 77) and use the same scripted model:

```python
def test_chat_stream_sends_the_draft_then_the_same_body_as_chat(indexed):
    client = _client(indexed, <the round-trip test's FakeChatModel>)
    with client.stream("POST", "/v1/chat/stream", json={"message": "임차권등기명령 신청서에 적어야 하는 사항은?"}) as response:
        events = [json.loads(line) for line in response.iter_lines() if line]
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert events[0] == {"type": "draft_reset"} and events[-1]["type"] == "response"
    body = events[-1]["body"]
    assert body["kind"] == "answer" and "trace" not in body
    assert "".join(e["text"] for e in events if e["type"] == "draft") == body["answer"]["analysis"]
```

**Step 2: Run, expect FAIL** (404).

**Step 3: Implement.** Move the shared start of `chat` into a helper, then add the endpoint:

```python
async def _turn(body: ChatRequest, request: Request):
    """The orchestrator, the conversation id and the pinned passages for one chat turn."""
    orchestrator = await run_in_threadpool(request.app.state.get_orchestrator)
    pinned = []
    if body.pinned_chunk_ids:
        stores = list(_stores(await run_in_threadpool(request.app.state.get_retriever)).values())
        pinned = await run_in_threadpool(pinned_chunks, stores, body.pinned_chunk_ids)
    return orchestrator, body.conversation_id or uuid.uuid4().hex, pinned


@router.post("/v1/chat", ...unchanged decorator...)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    orchestrator, conversation_id, pinned = await _turn(body, request)
    response = await run_in_threadpool(orchestrator.handle, ...unchanged...)
    ...unchanged...


@router.post("/v1/chat/stream")
async def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
    """/v1/chat as NDJSON: {"type": "draft", "text"} pieces of the analysis while the model writes it,
    {"type": "draft_reset"} before each new attempt, then {"type": "response", "body"} with the
    /v1/chat body, or {"type": "error"}. The draft has not been verified; only the body has."""
    orchestrator, conversation_id, pinned = await _turn(body, request)
    events: queue.Queue = queue.Queue()

    def draft(text):
        events.put({"type": "draft_reset"} if text is None else {"type": "draft", "text": text})

    def run():
        try:
            response = orchestrator.handle(body.message, conversation_id=conversation_id, user_documents=body.user_documents,
                                           pinned=pinned, on_draft=draft)
            if not body.include_trace:
                response = response.model_copy(update={"trace": None})
            events.put({"type": "response", "body": response.model_dump(mode="json", exclude_none=True)})
        except Exception:
            logger.exception("chat turn failed")
            events.put({"type": "error"})
        finally:
            events.put(None)

    threading.Thread(target=run, daemon=True).start()

    def lines():
        while (event := events.get()) is not None:
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson")
```
Imports: `json`, `logging`, `queue`, `threading`, `from fastapi.responses import StreamingResponse`, `logger = logging.getLogger(__name__)`. Starlette runs a sync iterator in its thread pool, so `events.get()` never blocks the event loop (the PR #20 problem).

Also add a check to the test that the body equals what `/v1/chat` returns for the same scripted answer, minus `conversation_id`. If `exclude_none` output differs from FastAPI's `response_model_exclude_none`, the test shows it.

**Step 4: Run, expect PASS**, then the backend suite.

**Step 5: Commit** `feat: /v1/chat/stream sends the answer draft as NDJSON`

---

### Task 5: Flask reads the stream and forwards the draft

**Files:**
- Modify: `webapp/legal_backend.py` (`ask`, module docstring)
- Modify: `webapp/app.py:387-437` (`_generate_legal_response_stream`)
- Test: `webapp/tests/test_legal_backend.py`, `test_chat_streaming.py`, `test_case_documents_in_chat.py`

**Step 1: Change the tests first.**

`test_legal_backend.py`:
- `served(handler)` now patches `legal_backend.httpx.stream` with `client.stream`.
- Add `def streamed(*events): return httpx.Response(200, content="".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events))`.
- Handlers that returned `httpx.Response(200, json=ANSWER)` now return `streamed({"type": "response", "body": ANSWER})`.
- Callers read `events = list(legal_backend.ask(...))` and the body as `events[-1]["body"]`.
- The URL expectation becomes `http://legal:8000/v1/chat/stream`.
- New tests:
  - Draft events come through in order before the response.
  - `streamed({"type": "error"})` raises `LegalBackendUnavailable`.
  - `streamed({"type": "draft", "text": "가"})` with no response line raises `LegalBackendUnavailable`.
- Keep the existing tests for an unreachable service, a timeout and HTTP 500, with `list(...)` around `ask`.

`test_case_documents_in_chat.py:26`: `list(legal_backend.ask(...))`, and its handler returns `streamed(...)` (import it from `tests.test_legal_backend`).

`test_chat_streaming.py`:
- Add `"draft"` and `"draft_reset"` to `FRONTEND_EVENT_TYPES`.
- In `collect`, the default mock returns `[{"type": "response", "body": <body>}]` instead of the body.
- Add a test where the mock returns `[{"type": "draft_reset"}, {"type": "draft", "text": "분석"}, {"type": "response", "body": backend_answer()}]`. The SSE stream must contain `{"type": "draft_reset"}` and `{"type": "draft", "content": "분석"}` before `final_response`, and the final response must be unchanged.
- Any test that passes its own `mock.Mock` as `backend` (search `collect(mock`) must return the list form too.

**Step 2: Run FLASK_TESTS, expect FAIL.**

**Step 3: Implement.**

`legal_backend.ask` becomes a generator:
```python
def ask(base_url, message, conversation_id, documents=(), pinned=()):
    """One chat turn, read as it streams. Yields the service's {"type": "draft"|"draft_reset"} events,
    then {"type": "response", "body": <the /v1/chat body>}.
    `documents` ... (keep the existing two lines)"""
    endpoint = base_url.rstrip("/") + "/v1/chat/stream"
    ...payload unchanged...
    try:
        # TIMEOUT now limits each read, not the whole turn; the service sends nothing until the
        # draft starts (analysis and search first), so it still has to cover that wait.
        with httpx.stream("POST", endpoint, json=payload, timeout=TIMEOUT) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if event["type"] == "error":
                    raise LegalBackendUnavailable("the service failed the turn")
                yield event
                if event["type"] == "response":
                    return
    except (httpx.HTTPError, ValueError, KeyError) as error:
        raise LegalBackendUnavailable(f"{type(error).__name__}: {error}") from error
    raise LegalBackendUnavailable("the stream ended without an answer")
```
Update the module docstring from "/v1/chat" to "/v1/chat/stream". `import json` if missing.

`app.py`, replace `body = legal_backend.ask(...)`:
```python
            body = None
            for event in legal_backend.ask(legal_rag_url, user_input, _legal_conversation_id(), documents,
                                           pinned=[case["chunk_id"]] if case else ()):
                if event["type"] == "response":
                    body = event["body"]
                elif event["type"] == "draft":
                    yield _sse({'type': 'draft', 'content': event["text"]})
                else:
                    yield _sse({'type': 'draft_reset'})
```
Rewrite the function docstring: the draft is forwarded unverified, and the browser shows it only as a labelled draft. Only the final body passes both checks and is stored. Keep the "no fallback to a local model" sentence.

**Step 4: Run FLASK_TESTS, expect PASS** (179 plus the new ones).

**Step 5: Commit** `feat: the web chat forwards the answer draft`

---

### Task 6: The browser shows the draft

**Files:**
- Modify: `webapp/static/lc-chat.js`

No JS test harness exists. Check it by hand in Task 7.

**Step 1: Implement.**
- In `P`, add `writing: '',   /* 답변 초안 — 검증 전. /chat draft 이벤트로 쌓이고 최종 답이 오면 사라진다 */`.
- `sendMessage`: set `P.writing = ''` with `P.thinking = true`.
- `sse.onmessage`:
  - `draft`: `P.writing += data.content; LC.render();`
  - `draft_reset`: `P.writing = ''; LC.render();`
  - `final_response`, `error`, `stream_end`, `onerror` and the `catch` branches: add `P.writing = '';`.
- In the `P.thinking` block of `main()`, put the draft above the status line when `P.writing` is set. Dim it, use the agent bubble shape, and add the label:
```js
      ${P.thinking && P.writing ? `
      <div style="display:flex;justify-content:flex-start"><div style="max-width:100%;padding:18px 22px;border-radius:5px 16px 16px 16px;border:1px dashed #e6e3dd;background:#faf9f6;color:#767d83;font-size:14px;line-height:1.85;white-space:pre-wrap;text-wrap:pretty">
        <div style="margin-bottom:8px;font-size:11px;letter-spacing:.08em;color:#9aa0a5">작성 중 · 검증 전</div>${esc(stripSourceIds(P.writing))}</div></div>` : ''}
```
- Add a helper near `sourceCards`:
```js
  /* 초안의 근거 표시(S1, D2)는 지운다. 최종 번호 [1]은 답이 끝나야 정해진다 */
  const stripSourceIds = text => text.replace(/\s*[(\[]?\s*(?:[SD]\d+\s*,?\s*)+[)\]]?/g, '');
```
Check that this regex leaves ordinary Korean text alone. It only matches an S or D followed by digits. Review it against a sample analysis from Task 7 and narrow it if it eats real text, e.g. "제3조 D동".

**Step 2: Commit** `feat: show the answer draft while it is written`

---

### Task 7: Live check by hand

**Step 1:** Run the backend and Flask suites. Both must pass.

**Step 2:** Stop any running backend. Start the API on 8011 and Flask on 5015 the same way `data/eval/reports/chat_pipeline/latency-e2e-2026-10-10/driver.sh` does, with scratch state. Use the `run` skill or Playwright to ask "전세 보증금을 못 돌려받고 있어요". Check:
- A dimmed draft labelled "작성 중 · 검증 전" grows.
- The verified answer replaces it, with the summary first and [1] numbers.
- A clarification turn shows no draft.

Take one screenshot of the draft for the user. Check the Ollama journal for `truncating input prompt` (with `--since`, never a full journal scan).

**Step 3:** Stop both servers.

---

### Task 8: Pre-registered latency and equality run

**Files (git-ignored):** `data/eval/reports/chat_pipeline/latency-stream-2026-10-17/`

**Step 1: Write `PREREG.txt` before running.**
- Same 158 turns as the `latency-e2e-2026-10-10` questions.
- (a) Time from sending a message to the first draft event: median and p90, warm and follow-up. Reported, with no pass bar.
- (b) Time to `final_response`: warm median must be ≤ the baseline 17.5 s + 1.0 s. The tolerance is there because the same question varies by a median of 5 s between passes.
- (c) Zero new errors.
- Answer equality: run 20 dev-80 questions through `Orchestrator.handle` twice each with `options.seed` fixed and temperature 0, once with `on_draft` and once without, and compare the answer JSON. First run the same question twice without streaming. If those two differ, seed does not make the output repeatable here; then report only GenerationError counts and tokens/s (from the `ollama_generation` log lines) for both arms. The seed goes in only for this script, by calling the model with an `options` override inside the script, not in served code.

**Step 2:** Copy `run.py`, `driver.sh` and `questions.json` from `latency-e2e-2026-10-10`. In `run.py` add `elif kind == "draft": row.setdefault("first_draft", now)`. In the report, add `line("   first draft", sel, "first_draft")` for warm and follow-up. In `driver.sh`, set `E` to the new folder and `S` to this session's scratchpad. Check `journalctl -u ollama --since today | grep -o "OLLAMA_FLASH_ATTENTION:[a-z]*" | tail -1` shows `true`.

**Step 3:** Run it through `four-levers-2026-10-09/gpu.sh` in the background (about an hour). Nothing else heavy may run in parallel. Then run the equality script.

**Step 4:** Write the results into `PREREG.txt` under "Results", against each criterion.

---

### Task 9: Record and hand off

- README:
  - Where it states the web-chat speed, add the first-draft time.
  - In section 7 limits, note that the draft is unverified and only the analysis.
- Add `docs/experiments/answer-streaming-2026-10-17.md` with the measured numbers, and link it from `docs/README.md`.
- Update the memory files `latency-e2e-baseline.md` and `chat-latency-cuts.md` ("streaming: tested").
- Ask the user before pushing or opening a PR.

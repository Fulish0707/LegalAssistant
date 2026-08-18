# Local Audio and Document Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add local STT and document analysis through the local multimodal backend while preserving existing routes, cleanup, identity isolation, and HTML sanitization.

**Architecture:** `LocalMultimodalService` owns all local inference and typed errors. `app.py` branches only for local audio and status mapping; `llm_ocr.py` retains extraction/prompt/sanitization and chooses the adapter in local mode. `document_vision.py` has no Flask or Ollama dependency and returns bounded base64 payloads.

**Tech Stack:** Flask, unittest/mock, requests, Pillow, PyPDF2, python-docx, faster-whisper, Ollama HTTP API.

---

### Task 1: Local multimodal service foundation

**Files:**
- Create: `local_multimodal.py`
- Modify: `app.py:1-45`
- Test: `tests/test_local_multimodal.py`

- [ ] **Step 1: Write failing adapter tests**

```python
def test_analyze_document_posts_text_and_images_to_ollama(self):
    with patch("local_multimodal.requests.post") as post:
        post.return_value.json.return_value = {"message": {"content": "report"}}
        result = self.service.analyze_document("Analyze this document", ["base64-data"])
    self.assertEqual(result, "report")
    post.assert_called_once_with(
        f"{self.config.ollama_url}/api/chat",
        json={"model": self.config.model, "messages": [{"role": "user", "content": "Analyze this document", "images": ["base64-data"]}], "stream": False},
        timeout=self.config.request_timeout,
    )

def test_analyze_document_raises_typed_error_for_malformed_response(self):
    with patch("local_multimodal.requests.post") as post:
        post.return_value.json.return_value = {"message": {}}
        with self.assertRaises(LocalInferenceUnavailableError):
            self.service.analyze_document("Analyze this document")
```

- [ ] **Step 2: Run focused adapter tests and verify RED**

Run: `python -m unittest tests.test_local_multimodal -v`

Expected: FAIL because `local_multimodal` and `analyze_document` do not exist.

- [ ] **Step 3: Implement config, typed errors, semaphore, STT, and document adapter**

```python
class LocalInferenceError(RuntimeError):
    pass

class LocalInferenceUnavailableError(LocalInferenceError):
    pass

def analyze_document(self, prompt, images=None):
    message = {"role": "user", "content": prompt}
    if images:
        message["images"] = images
    return self._post_chat([message])
```

Implement `_post_chat` once, with the process-wide semaphore around `requests.post`, a payload of `model`, `messages`, and `stream=False`, `raise_for_status()`, and strict extraction of non-empty `message.content`. Convert request, HTTP, JSON, and malformed-content failures into `LocalInferenceUnavailableError`. Implement `generate(messages)` via the same helper. Implement `transcribe_audio(path, language="ko")` with lazy faster-whisper loading and typed errors; do not load it at module import.

- [ ] **Step 4: Run focused adapter tests and verify GREEN**

Run: `python -m unittest tests.test_local_multimodal -v`

Expected: PASS with no real HTTP request.

### Task 2: Local audio route contract

**Files:**
- Modify: `tests/test_app_contract.py`
- Modify: `app.py:1-45,746-930`

- [ ] **Step 1: Write failing local audio route tests**

```python
def test_local_audio_uses_local_stt_and_local_summary(self):
    service.transcribe_audio.return_value = {"transcript": "로컬 전사 결과", "segments": [{"start": 0.0, "end": 1.25, "text": "로컬 전사 결과"}], "language": "ko"}
    service.generate.return_value = "로컬 요약"
    response = self.client.post("/transcribe_and_summarize_audio_file", data={"audio_file": (io.BytesIO(b"RIFF"), "sample.wav")}, content_type="multipart/form-data")
    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.json["transcript"], "로컬 전사 결과")
    self.assertEqual(response.json["summary"], "로컬 요약")
    self.assertEqual(response.json["segments"][0]["end"], 1.25)
    self.assertIn("로컬 전사 결과", service.generate.call_args.args[0][0]["content"])
    post.assert_not_called()
```

Add separate transcription and summary typed-error tests. Each posts the same valid upload, asserts status 503, asserts the fixed public error contains `Local audio processing is unavailable`, asserts it excludes `private implementation detail`, and compares the upload folder contents before and after response. Add environment snapshot/restore helpers that restore `GOOGLE_API_KEY`, `client_id`, `client_secret`, `MODEL_BACKEND`, and any Naver aliases after every test, then reload `app` after setting `MODEL_BACKEND=local`.

- [ ] **Step 2: Run contract tests and verify RED**

Run: `python -m unittest discover -s tests -p 'test_app_contract.py' -v`

Expected: FAIL because credentials/network STT run before local inference.

- [ ] **Step 3: Add the minimal local branch and route 503 mapping**

```python
config = LocalMultimodalConfig.from_environment()
if config.backend == "local":
    try:
        local_result = LocalMultimodalService(config=config).transcribe_audio(audio_file_path, language="ko")
        transcript = local_result.get("transcript", "")
        base = {"transcript": transcript, "segments": local_result.get("segments", []), "language": local_result.get("language", "ko")}
        if not transcript.strip():
            return {**base, "summary": "Audio appears to be silent or could not be transcribed."}
        summary = LocalMultimodalService(config=config).generate([{"role": "user", "content": build_korean_audio_summary_prompt(transcript)}])
        return {**base, "summary": summary}
    except LocalInferenceError:
        return {"error": LOCAL_AUDIO_UNAVAILABLE_MESSAGE, "transcript": "", "summary": "", "local_inference_unavailable": True}
```

Reuse one service instance within the branch. Extract the shared Korean summary text to `build_korean_audio_summary_prompt` so the existing Naver/Gemini branch uses exactly the same instruction. In both file and blob routes, map `local_inference_unavailable` to HTTP 503 before existing error matching. Do not alter Naver request or Gemini invocation behavior.

- [ ] **Step 4: Run audio-focused and complete contract tests**

Run: `python -m unittest discover -s tests -p 'test_app_contract.py' -v`

Expected: PASS; mocked `requests.post` is never called in local audio cases.

- [ ] **Step 5: Commit local audio integration**

```bash
git add app.py local_multimodal.py tests/test_app_contract.py tests/test_local_multimodal.py
git commit -m "feat: route audio through local multimodal backend"
```

### Task 3: Local TXT/DOCX document analysis

**Files:**
- Modify: `tests/test_app_contract.py`
- Modify: `app.py:637-710`
- Modify: `llm_ocr.py:222-268`

- [ ] **Step 1: Write the failing local document route test**

```python
def test_local_text_document_uses_adapter_sanitizes_and_ingests(self):
    service.analyze_document.return_value = "<script>bad()</script><h2>Safe</h2>"
    response = self.client.post("/process_document_ocr", data={"file": (io.BytesIO(b"normalized source"), "sample.txt"), "analysis_mode": "summary", "user_instructions": "Focus on risks", "add_to_knowledge_base": "true"}, content_type="multipart/form-data")
    self.assertEqual(response.status_code, 200)
    self.assertIn("Safe", response.json["report_html"])
    self.assertNotIn("script", response.json["report_html"].lower())
    self.assertEqual(captured["text"], "normalized source")
    self.assertTrue(captured["user_id"])
    self.assertIn("Document Summary", service.analyze_document.call_args.args[0])
    self.assertIn("Focus on risks", service.analyze_document.call_args.args[0])
```

Set `MODEL_BACKEND=local`, reload `app`, mock `app.LocalMultimodalService`, and mock `app.ingest_text_to_vector_store`. The test must demonstrate local mode works without `GOOGLE_API_KEY` and does not require an OCR dependency for TXT/DOCX.

- [ ] **Step 2: Run the contract test and verify RED**

Run: `python -m unittest discover -s tests -p 'test_app_contract.py' -v`

Expected: FAIL because `/process_document_ocr` rejects `llm is None` before local handling.

- [ ] **Step 3: Implement narrow local document routing**

```python
if LocalMultimodalConfig.from_environment().backend == "local":
    report_html = LocalMultimodalService(config=config).analyze_document(report_generation_input)
else:
    formatted_prompt_messages = prompt_template.format_messages(...)
    ai_response = llm.invoke(formatted_prompt_messages)
    report_html = ai_response.content if hasattr(ai_response, "content") else str(ai_response)
report_html = clean_html_response(report_html)
```

Pass `local_config` and/or `local_service` explicitly into `llm_ocr.analyze_document` so the route owns app-boundary construction and tests can mock it there. Alter the route prerequisites to permit local TXT/DOCX analysis without a Google LLM, while retaining OCR prerequisite checks for image/PDF input. Preserve `include_extracted_text` and the existing route’s ingestion with `session["user_id"]`.

- [ ] **Step 4: Run focused and full contract tests**

Run: `python -m unittest discover -s tests -p 'test_app_contract.py' -v`

Expected: PASS; the response is sanitized and ingestion receives the active anonymous user id.

- [ ] **Step 5: Commit local text documents**

```bash
git add app.py llm_ocr.py tests/test_app_contract.py
git commit -m "feat: analyze text documents with local backend"
```

### Task 4: Bounded visual-document payloads

**Files:**
- Create: `document_vision.py`
- Create: `tests/test_document_vision.py`
- Modify: `app.py`
- Modify: `llm_ocr.py`

- [ ] **Step 1: Write failing vision-helper tests**

```python
def test_image_payload_returns_one_base64_image(self):
    self.assertEqual(len(build_local_document_images(self.image_path, "png")), 1)

def test_pdf_payloads_stop_at_four_pages(self):
    self.assertEqual(len(build_local_document_images(self.pdf_path, "pdf")), 4)

def test_oversized_image_is_resized_with_aspect_ratio(self):
    payload = build_local_document_images(self.large_image_path, "png")[0]
    self.assertLessEqual(decoded_image(payload).width, 1600)
    self.assertEqual(decoded_image(payload).height, 800)

def test_conversion_failure_is_safe(self):
    with self.assertRaises(DocumentVisionError) as raised:
        build_local_document_images("/private/input.pdf", "pdf")
    self.assertNotIn("/private/input.pdf", str(raised.exception))
```

- [ ] **Step 2: Run helper tests and verify RED**

Run: `python -m unittest tests.test_document_vision -v`

Expected: FAIL because `document_vision` does not exist.

- [ ] **Step 3: Implement pure bounded vision conversion**

```python
MAX_LOCAL_DOCUMENT_PAGES = 4
MAX_LOCAL_IMAGE_EDGE = 1600

def build_local_document_images(file_path, file_extension):
    if file_extension in {"png", "jpg", "jpeg"}:
        return [_encode_resized_image(Image.open(file_path))]
    if file_extension == "pdf":
        return [_encode_resized_image(page) for page in render_pdf_pages(file_path, limit=MAX_LOCAL_DOCUMENT_PAGES)]
    return []
```

Use a supported PDF renderer installed as an explicit dependency. Resize with Pillow `thumbnail((1600, 1600))`, encode PNG bytes, and return bare base64 strings. Catch renderer/image exceptions and raise `DocumentVisionError("Document conversion failed. Upload a valid image or PDF.")` only.

- [ ] **Step 4: Connect visual payloads only in local mode**

```python
try:
    images = build_local_document_images(file_path, file_extension)
except DocumentVisionError as exc:
    return {"error": str(exc)}, 400
report_html = local_service.analyze_document(report_generation_input, images=images)
```

Keep extraction and `clean_html_response` unchanged. Do not use conversion for non-local document analysis or TXT/DOCX. Add route tests asserting image payloads reach the adapter and safe conversion errors become HTTP 400.

- [ ] **Step 5: Run helper, route, and full tests**

Run: `python -m unittest tests.test_document_vision -v && python -m unittest discover -s tests -v`

Expected: PASS with no external local inference or model calls.

- [ ] **Step 6: Commit bounded local visual documents**

```bash
git add document_vision.py tests/test_document_vision.py app.py llm_ocr.py requirements.txt
git commit -m "feat: bound local visual document analysis"
```

### Task 5: Local runtime documentation and release verification

**Files:**
- Modify: `.env.example`
- Modify: `README.md`

- [ ] **Step 1: Document supported settings**

```dotenv
MODEL_BACKEND=local
OLLAMA_MODEL=gemma3n:e4b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
STT_MODEL=small
LOCAL_STT_DEVICE=cuda
LOCAL_STT_COMPUTE_TYPE=int8_float16
LOCAL_GPU_CONCURRENCY=1
```

Explain that the first STT request downloads and loads the Whisper model and that 8 GB hardware should keep local GPU concurrency at one.

- [ ] **Step 2: Run final verification**

Run: `git diff --check && python -m unittest discover -s tests -v && MODEL_BACKEND=local OLLAMA_MODEL=gemma3n:e4b python -c "import app; print(app.app.test_client().get('/health').json)"`

Expected: clean diff, all tests pass, and health includes local backend/Ollama/model/STT readiness without invoking inference.

- [ ] **Step 3: Commit documentation**

```bash
git add .env.example README.md
git commit -m "docs: describe local multimodal configuration"
```

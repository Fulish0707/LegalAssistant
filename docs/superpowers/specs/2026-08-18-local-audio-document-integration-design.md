# Local Audio and Document Integration Design

## Goal

Route local audio, text-document analysis, and later bounded visual-document analysis through a small local multimodal adapter without changing public URLs, anonymous-user isolation, upload cleanup, or server-side HTML sanitization.

## Architecture

`LocalMultimodalService` is the sole boundary for local inference.  Its audio
contract exposes `transcribe_audio(path, language)` and its document contract
exposes `analyze_document(prompt, images=None)`.  The service converts expected
local-runtime failures to existing typed local-inference errors and uses its
process-wide inference semaphore for document requests as it does for text
generation.

`app.process_audio_common` reads `LocalMultimodalConfig` first.  With the local
backend selected, it calls local STT and then local generation with the existing
Korean professional-summary prompt.  Typed local-inference errors become a
generic action-oriented error payload that route code maps to HTTP 503.  The
existing Naver/Gemini path is unchanged.

For TXT and DOCX uploads, `llm_ocr.analyze_document` retains extraction and the
existing prompt construction.  In local mode it sends the resulting prompt to
the adapter, then always sends returned HTML through `clean_html_response`.
Knowledge-base ingestion continues to receive normalized extracted text and the
active user id through the existing route flow.

Visual PDF/image input is deferred until text-document routing is green.  A
Flask- and Ollama-independent `document_vision.py` converts at most four PDF
pages or one image into base64 payloads, limits the longest image edge to 1600
pixels while retaining aspect ratio, and reports conversion errors without
paths or internal details.  Only local document analysis uses this helper.

## Error Handling and Compatibility

- Local audio typed failures return HTTP 503 with the generic local-runtime
  action message; implementation details never reach clients.
- Conversion failures return a safe HTTP 400 without exposing a source path.
- Empty local transcripts use the current silent-audio behavior and preserve
  STT segments and language metadata.
- Existing URLs, temporary-file cleanup, anonymous identity boundaries, and
  sanitization behavior remain unchanged.

## Test Strategy

Use red-green TDD.  App contract tests mock `LocalMultimodalService` at the
`app` module boundary and assert that local audio never invokes Naver requests,
that typed failure responses are safe and cleanup occurs, and that local
TXT/DOCX reports are sanitized and ingested with the active identity.

Adapter unit tests mock `requests.post` and assert the Ollama `/api/chat`
payload, returned content, and malformed-response typed errors.  Vision-helper
unit tests cover image payload creation, four-page bounds, aspect-ratio resize,
and safe conversion failure.  Tests restore `GOOGLE_API_KEY`, Naver settings,
and `MODEL_BACKEND` after every run; no real Whisper, model download, or Ollama
call is permitted.

## Delivery Sequence

1. Local audio contract and safe failures; implement the minimal local branch.
2. Document adapter contract; implement it using the shared semaphore.
3. Local TXT/DOCX routing and sanitization.
4. Bounded local visual document handling.
5. Environment documentation and complete verification.

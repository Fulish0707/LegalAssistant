# Upload and Expensive-Route Hardening Design

## Goal

Protect upload and AI-backed routes from oversized payloads, mislabeled files, resource-exhaustion inputs, and request bursts while preserving the existing API response format where requests are valid.

## Scope

The first hardening slice covers statistics, OCR/document, and audio uploads, plus expensive chat/AI routes. It does not add infrastructure such as Redis, a reverse proxy, or asynchronous job processing.

## Architecture

Create a focused `security.py` module with two independent responsibilities:

1. Upload validation applies a 25 MiB request cap, extension policy, lightweight file-signature checks, and format-specific complexity limits before application routes save a file or invoke a parser.
2. Burst limiting persists fixed one-minute windows in SQLite. The key is the authenticated account ID when present, otherwise the salted hash of the normalized client IP already used by quota handling. Each key may make at most 10 expensive requests per minute.

The SQLite table is created by the existing schema initializer. Atomic upserts ensure concurrent requests cannot exceed the limit. Expired rows may remain initially; bounded cleanup is deferred to production operations work.

## Upload policy

Flask rejects request bodies exceeding 25 MiB with JSON `413` responses. Each route additionally validates the uploaded stream before saving it:

- OCR documents: PNG/JPEG signatures, PDF `%PDF-` header, DOCX ZIP container, or plain-text UTF-8 decoding.
- Statistics: UTF-8 CSV/TSV content or XLSX ZIP container.
- Audio: WAV RIFF/WAVE header, OGG header, FLAC header, MP3 ID3/frame sync, MP4-family `ftyp` marker for M4A, and WebM EBML header.

Complexity checks use safe metadata available without full AI processing: PDF page count, DOCX/XLSX ZIP member count and uncompressed size, CSV/TSV row and column limits, and WAV duration. Unsupported, malformed, or over-complex uploads receive JSON `415` or `422` errors and are never persisted.

## Route behavior

Routes validate file size and content before creating a temporary file. After validation succeeds, they reserve the burst-limit slot immediately before their costly parsing or provider invocation. Upload endpoints and guest chat both use the 10-per-minute limit; account routes are keyed by account, while guest chat is keyed by salted client IP.

Error responses are stable and non-sensitive:

- `413 payload_too_large`
- `415 unsupported_media_type`
- `422 document_too_complex`
- `429 rate_limit_exceeded`, including the limit and retry window

Existing daily AI quotas continue to apply independently after burst validation.

## Testing

Tests cover global size handling, signature mismatches, valid representative signatures, document-complexity rejections, per-account and per-guest-IP rate isolation, concurrent reservations, and route-level proof that rejected uploads do not reach temporary storage, parsers, or providers.

## Non-goals

This stage does not introduce malware scanning, content-disarm-and-reconstruct, Redis/distributed limiting, background jobs, object storage, or document-level RAG deletion. Those require separate infrastructure or lifecycle decisions.

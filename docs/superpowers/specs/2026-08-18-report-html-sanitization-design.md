# Report HTML Sanitization Design

## Goal

Prevent LLM-generated document and statistics reports from executing browser code while preserving useful report structure.

## Approach

Sanitize report HTML on the server with Bleach immediately before it is returned in API JSON. Client templates may continue rendering the trusted, server-sanitized result with `innerHTML`; no caller can bypass the API boundary by supplying a different client.

## Allowed Output

The sanitizer preserves headings, paragraphs, line breaks, emphasis, block quotes, preformatted/code text, ordered and unordered lists, tables, and links. Links retain only `href`, `title`, and `target` attributes and allow only `http`, `https`, and `mailto` URLs. Tables retain structural attributes needed for accessible report markup.

All styles, images, forms, scripts, embedded content, event-handler attributes, data URLs, JavaScript URLs, and unrecognized elements are removed. Link targets receive `rel="noopener noreferrer"` when opened in a new tab.

## Integration

`sanitize_report_html` lives in a focused new module and is called by `llm_ocr.clean_html_response` and by the statistics route before assigning AI insight to the rendered report. Locally generated statistics tables remain valid under the same sanitizer path.

## Tests

Tests verify script tags, inline handlers, dangerous URLs, iframes, and styles are removed. They also verify valid headings, lists, tables, and HTTPS links remain usable. Route-level tests confirm both document and statistics report responses use the sanitizer.

## Acceptance Criteria

- No API-produced report contains executable HTML or unsafe URL schemes.
- Valid report headings, lists, tables, and safe links survive.
- Existing report rendering behavior remains intact.
- The full test suite passes.

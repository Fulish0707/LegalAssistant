# Report HTML Sanitization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure every AI-produced report is safe for the existing `innerHTML` renderers.

**Architecture:** A focused server-side sanitizer allowlists report markup before OCR and statistics APIs return it. Templates remain unchanged because they receive only sanitized HTML.

**Tech Stack:** Python, Bleach, Flask, `unittest`.

---

### Task 1: Add the allowlist sanitizer

**Files:**

- Create: `html_sanitizer.py`
- Modify: `requirements.txt`
- Test: `tests/test_html_sanitizer.py`

- [ ] **Step 1: Write failing tests that remove `<script>`, `onerror`, `javascript:` URLs, iframes, and style attributes while preserving a heading, list, table, and HTTPS link.**
- [ ] **Step 2: Run `python -m unittest discover -s tests -p 'test_html_sanitizer.py' -v`; expect import failure.**
- [ ] **Step 3: Add `bleach` to requirements and implement `sanitize_report_html(value)` with explicit tags, attributes, and URL protocols.**
- [ ] **Step 4: Re-run the focused tests; expect PASS.**
- [ ] **Step 5: Commit with `git add html_sanitizer.py requirements.txt tests/test_html_sanitizer.py && git commit -m "feat: sanitize generated report HTML"`.**

### Task 2: Sanitize every report response

**Files:**

- Modify: `llm_ocr.py:129-166`
- Modify: `app.py:583-606`
- Test: `tests/test_app_contract.py`

- [ ] **Step 1: Write failing route tests that mock an LLM response containing unsafe HTML and assert the returned `report_html` excludes it.**
- [ ] **Step 2: Run the tests; expect failure because unsafe output is returned unchanged.**
- [ ] **Step 3: Call `sanitize_report_html` at the end of OCR cleanup and before statistics HTML rendering.**
- [ ] **Step 4: Re-run focused tests, then `python -m unittest discover -s tests -v`; expect PASS.**
- [ ] **Step 5: Commit with `git add app.py llm_ocr.py tests/test_app_contract.py && git commit -m "feat: sanitize report API responses"`.**

### Task 3: Verify and document behavior

**Files:**

- Modify: `README.md`

- [ ] **Step 1: Document that generated report markup is sanitized and which structural formatting remains available.**
- [ ] **Step 2: Run `git diff --check` and `python -m unittest discover -s tests -v`; expect zero failures.**
- [ ] **Step 3: Commit with `git add README.md && git commit -m "docs: document report HTML sanitization"`.**

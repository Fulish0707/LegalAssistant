# Relevance judge repair implementation plan

**Goal:** Repair quote extraction and clarify evidence roles, then compare original and revised judgments on fresh IP questions.

**Architecture:** Keep the existing 0/1/2 schema and append-only judgment store. Add an index-only, source-copy quote retry. Freeze original code and reference artifacts to compare separate immutable judge keys.

**Tech stack:** Python, Pydantic, existing Ollama Qwen3:8b client and calibration/reporting modules.

## Steps

- [x] Inspect initial calibration failures, current code and working-tree baseline.
- [x] Save original judge source and baseline metadata in the experiment artifact directory.
- [x] Draw question- and gold-document-disjoint 30-pair calibration plus six declared negative controls; freeze AI reference labels before model calls.
- [x] Add one source-span quote-selection retry, original-quote audit fields and hash versioning in `src/legal_rag/evaluation/relevance_judge.py`.
- [x] Clarify party-assertion and procedure/scope boundaries in the relevance system prompt.
- [x] Run original and revised local judges against the same fresh calibration using their distinct keys.
- [x] Compare agreement, quote failures, repaired spans and paired outcomes; review disagreements.
- [x] Verify reference hashes and focused diff; save reproducible commands and report limitations.

Result: both 36/36 resolved. Exact agreement unchanged at 61.1%; broad AC1 0.876 to 0.919; direct-basis AC1 0.389 to 0.341. Six source-copy retries succeeded mechanically, but four recovered quotes did not establish direct support under frozen references. The repair is not a demonstrated answer-quality or grade-2 calibration improvement. Detailed evidence is in `data/eval/reports/structural-all-2026-10-03/judge-repair/`.

Scope excludes unrelated files, original QA/gold labels, indexes, retrieval profiles, unit-test suites, commits and publishing. Execute inline; user approval already covers this plan.

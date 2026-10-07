# Corpus-specific prompt pilot implementation plan

**Goal:** Compare the shared relevance prompt with corpus guidance on identical frozen examples across four corpora.

**Architecture:** Isolated experiment scripts load a frozen judge module and change only SYSTEM for treatment. Existing append-only keys and reporting helpers preserve provenance. Execute inline in the current workspace, preserving unrelated edits; no commits or production default changes.

**Tech stack:** Python, existing Ollama client, relevance judge, judgment store and calibration helpers.

- [x] Snapshot current judge; define and freeze four guidance blocks before reading outcomes.
- [x] Seed and copy 12 references per corpus and matching passages; freeze hashes, selection and source provenance.
- [x] Implement a resumable runner for both variants and per-corpus reporting.
- [x] Run both variants with Qwen3:8b thinking on and identical settings; record digest and judgments.
- [x] Review changed grades and repaired quotes, verify all frozen hashes, and report measured effects and limits.

All 96 judgments resolved. Exact agreement: shared 33/48, specialized 36/48; seven wins, four losses. Civil improved 6/12 to 9/12, administrative 10/12 to 11/12, criminal regressed 8/12 to 7/12, IP stayed 9/12. Source audit found imperfect reasons even for some nominal wins, corpus/substantive-domain mismatches and one repeated shared-control variation. No prompt was promoted. Civil is a candidate for a larger repeat evaluation with human-calibrated references.

Artifacts: `data/eval/reports/domain-prompt-pilot-2026-10-03/`. Direct authorized calibration is the validation; no unit-test suites, indexing, labeling of original datasets, answer-generation changes or publishing are in scope.

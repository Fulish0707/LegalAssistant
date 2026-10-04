# Golden-answer comparison plan

**Goal:** Measure whether smaller structural chunks improve generated answers.

**Architecture:** A resumable evaluation entrypoint uses the existing retrieval,
context, Gemini replay and v2 answer-grading components. Run inline; preserve other
agents' work and existing artifacts. No commits or unit-suite changes.

- [x] Add `scripts/eval_golden_answers.py` with prepare, generate, judge and report phases.
- [x] Freeze the historical 80 question/reference rows and both arms' retrieval settings.
- [x] Retrieve both arms, save full evidence bundles and verify budgets.
- [x] Generate fresh Gemini answers in alternating arm order; save full text and usage.
- [x] Grade both arms with identical v2 settings; retry unresolved calls using existing policy.
- [ ] Validate frozen hashes, report paired/per-group scores and review changed cases.

Run all phases: `PYTHONPATH=src .venv/bin/python scripts/eval_golden_answers.py all`.
Each named phase is independently resumable under the same default output directory.

Completed 2026-10-04: both arms80/80. Generated comparison and all-case review export; human review of changed cases remains outstanding. Grounded agreement38/80→56/80, paired interval+11.25 to+33.75pp.

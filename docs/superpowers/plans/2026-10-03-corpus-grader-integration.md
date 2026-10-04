# Corpus grader integration implementation plan

**Goal:** Apply corpus guidance to main grading commands and evaluate a criminal revision.

**Architecture:** Central prompt policy, backward-compatible shared hash, question metadata routing, corpus-family keys and matching CLI selectors. Execute inline in the existing workspace.

- [x] Archive old judging/agreement entrypoints used by frozen reports.
- [x] Add central guidance and question-domain metadata routing; preserve domain in new pools.
- [x] Integrate profile/domain choices, audit fields and hash versioning in judge and six relevant CLIs.
- [x] Verify CLI selection/routing and archive frozen criminal references before model calls.
- [x] Evaluate revised criminal guidance through the main CLI, compare with frozen pilot results and review changes.
- [x] Document runnable commands, verified outcomes and limitations; preserve unrelated work.

Validation consists of CLI/import/routing checks and the requested live calibration. No unit-test suite, commits, indexing or generation-prompt modifications are included.

Outcome: criminal revision 7/12, shared 8/12; default activates civil, administrative and IP only. Criminal candidate remains opt-in via `corpus-experimental`.

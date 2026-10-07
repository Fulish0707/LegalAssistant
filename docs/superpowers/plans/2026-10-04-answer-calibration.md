# Answer calibration implementation plan

**Goal:** Address the documented grading and claim-attribution failures while retaining historical evaluation provenance.

**Architecture:** A versioned answer-v3 module shares the existing quote and metric primitives. The saved-answer CLI selects the rubric module and separate artifact paths. Structural chunk roles propagate to retrieval text and context headers.

## Steps

- [x] Add v3 coverage/basis checklists, prompt counterexamples and validators in `src/legal_rag/evaluation/answer_judge_v3.py`.
- [x] Extend `scripts/judge_answers.py` to select v3, retry literal span errors and read versioned reports without changing v2 defaults.
- [x] Add explicit-heading role classification; annotate metadata/headers and preserve role boundaries in `src/legal_rag/chunking/imported.py` and `korean_structure.py`.
- [x] Render roles in `src/legal_rag/rag/context.py` and add source-attribution instructions in the shared answer prompt.
- [x] Add a saved-answer paired v3 comparison command and document exact runnable commands and indexing implications.
- [x] Inspect diffs and syntax; record that model calibration and tests have not been run.

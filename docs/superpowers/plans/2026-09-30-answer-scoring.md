# Answer scoring implementation plan

**Goal:** Apply the approved separate scoring dimensions consistently to both models.

**Architecture:** A reusable v2 rubric validates source quotes and derives metrics.
The existing CLI retains its legacy mode and uses separate versioned artifacts for
the new rubric. Input fingerprints and immutable manifests protect resumed work.

**Tech stack:** Python, Pydantic, Typer, the existing Ollama chat adapter.

- [x] Add the v2 schema, explicit Korean rubric, quote validation and metric functions
  to src/legal_rag/evaluation/answer_judge.py.
- [x] Wire evidence loading, immutable manifests, successful-only resumption and
  separate reports into scripts/judge_answers.py. Preserve legacy constants used by
  existing experiment scripts.
- [x] Document v2 commands, metric meanings and calibration limits. Review the diff
  and syntax without running tests or invoking an LLM.

## Grader repair

- [x] Collect all rejected quote fields and supply their failures to the second attempt.
- [x] Persist raw/parsed attempts, feedback, validation issues and available generation metadata.
- [x] Record supplemental retry provenance, preserve successful grades, and disclose policy counts.
- [x] Complete static review, syntax checks and documentation; leave live grading to the user's restart.

## Passage-selection repair

- [x] Inspect rejected drafts showing paraphrases after feedback retries.
- [x] Build exact span catalogues with constrained reference/answer/evidence ID choices.
- [x] Materialize IDs into the existing grade schema, preserving strict quote checks and valid grades.
- [x] Record policy/schema/catalogue provenance; prepare actual failed-case requests and complete static review.
- [x] Document restart and the absence of live error-rate measurements for the new policy.

## Bounded selection and timeout recovery (2026-10-01)

- Cap evidence IDs at eight; enforce uniqueness in schema and Python validation.
- Version supplemental retry provenance as quote-selection-v3.
- Set request timeout to 180 seconds; do not immediately retry a timed-out request.
- Persist diagnostics/results and exit nonzero after two consecutive timeout cases.
- Preserve selection mode when resuming historical failed selection attempts.
- Inspect local clients before unloading the stalled judge worker.
- Input truncation observed in Ollama logs remains a separate evaluation limitation.

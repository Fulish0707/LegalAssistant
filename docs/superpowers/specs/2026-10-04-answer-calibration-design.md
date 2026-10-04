# Answer grading and evidence-role calibration

The user approved moving from the changed-case review to grader and chunk-role repairs.

## Design

Use a separate answer-v3 rubric. Keep v2 and legacy artifacts readable and unchanged. Require a checklist of core reference conclusions/conditions and separate named-basis checks with literal reference/answer spans. Validate spans and summary/checklist consistency; do not claim these checks prove semantic correctness. Evidence support remains independent of the reference. Add counterexamples for extra details, missing citations, supported abstention and party statements. Invalid judgments remain unresolved and retry with actionable feedback.

Add section_role to chunk metadata and rendered case evidence. Derive roles only from explicit structural headings, retain unknown for flattened inputs, and prevent packing or merging across role boundaries. Chunk headers carry the role so reranking and generation see it. Preserve source text and existing corpora; the change applies on future rechunking. Warn generation against attributing party/unknown passages as established findings.

## Completion boundaries

Prepare runnable v3 regrading and comparison commands against the saved 160 answers; no answer regeneration or full indexing. No new tests or test runs unless asked. Inspect diffs and syntax. No commits or Notion publishing. Calibration gains require actual regrading and independent human labels; no predicted score is reported.

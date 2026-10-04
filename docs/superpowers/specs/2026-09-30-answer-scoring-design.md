# Consistent answer scoring

The approved change separates reference-conclusion coverage, legal-basis agreement,
and support from the evidence actually supplied to the generator. A different or
missing statute does not change conclusion coverage unless it changes the substantive
rule, but it does affect the separate legal-basis grade. Evidence absence remains
unknown rather than unsupported. Quoted justifications must exist in the inputs.

The legacy rubric and artifacts remain available. The default CLI uses version v2,
distinct output files and an immutable manifest including prompt, schema, model
digest and decoding settings. Resume uses fingerprints of question, reference,
answer and evidence; judge failures retry. Full source inputs are retained for review.
The report separates valid grades, empty generated answers and unresolved judge
errors, and prints conclusion, basis and grounded-answer metrics separately.

Implementation scope is scripts/judge_answers.py plus a reusable evaluation rubric
module and usage documentation. This change makes scoring rules explicit; human
calibration is still required before trusting new comparisons. No historical metrics
are revised by this implementation. No tests are added or run without a user request.

## Approved grader repair

Collect all quote failures with field names, rejected text and exact matching source
indexes. The second attempt receives the prior draft plus explicit correction
feedback while retaining the original inputs and grading rules. Each attempt saves
its raw output, parsed grade, error, feedback and available token/finish metadata.
The original rubric manifest stays immutable. A supplemental retry-policy manifest
and per-record hashes expose the changed retry strategy without discarding existing
valid grades. Reports validate this provenance and display policy counts.

## Passage-selection repair, 2026-10-01

Observed feedback-only retries still rewrite source text. The next retry policy
constrains reference/answer/evidence selections to IDs of unchanged source spans.
Code materializes their exact text into the existing v2 grade schema and applies
the same quote validation. No fuzzy matching or quote substitution is used. A_NONE
is an explicit empty-answer choice, permitted only with no coverage. Known failures
start with selection; new quote/schema failures switch their second attempt to it.
Semantic grades remain the judge's responsibility and need independent calibration.

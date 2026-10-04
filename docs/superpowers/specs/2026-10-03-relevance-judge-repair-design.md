# Structural IP relevance judge repair

User authorized the two fixes and fresh calibration with “proceed” after the initial 60-pair diagnostic round.

## Changes

1. Add explicit party-assertion and procedural-scope instructions. An unadopted assertion is at most partial relevance; distinguish pending-application amendments, post-registration corrections, domestic and international procedures. Preserve the distinction between related facts (1) and a different substantive issue (0). A reference answer is a fallible hint.
2. On grade 2 with an invalid quote, make one bounded quote-only retry. Supply exact source spans with indexes; require an index or zero for no support. Copy a selected span from source bytes; verify it with the unchanged presence check. Do not let retry regrade, accept paraphrases or silently upgrade grade 0/1. Retain original quote, original verification, selection and retry status. Failure retains the old downgrade to 1.
3. Change the judgment hash to include all retry prompts/schema/settings/span version and increment the quote-rule version. Existing judgments are immutable and coexist with the new key.

## Evaluation

Freeze the original judge source. Draw 30 pairs across 10 fresh questions from existing structural traces, excluding initial calibration questions, their normalized text and gold document IDs. Include candidate misses, fusion losses and the two remaining successful questions; no question-disjoint reranker-loss cases remain because all three were in development. Label before either judge runs. Reference review is AI source-text judgment, not human validation or current-law verification.

Compare original and revised judges on the same fresh sample. Fixed primary threshold: relevance boundary AC1 >=0.6; report explicit-rule boundary, exact agreement, model grades before quote checks, original invalid quotes, repair outcomes and role errors. Also report paired changes. Question clusters, development selection and limited sample size constrain conclusions. No full benchmark relabeling or production retrieval changes.

Before reference freeze and any model calls, natural reference review found no grade-0 pairs. Six explicitly marked cross-question unrelated controls were added, bringing the comparison to 36 pairs. Natural and control results must be reported separately. Controls have no retrieval rank and do not estimate retrieval precision.

## Completion

Implement bounded code changes, run the authorized calibration for both keys, review all disagreements and repaired quotes, verify frozen hashes and report actual metrics. Preserve unresolved outcomes and avoid claiming broader answer-quality improvement. No unit-test suite or commits are included in this task.

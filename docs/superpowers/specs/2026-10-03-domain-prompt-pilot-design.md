# Corpus-specific relevance prompt pilot

User approved testing the proposed shared grading rubric with domain-specific guidance. Scope is relevance grading, not answer generation.

Use the current revised shared relevance prompt as control. Treatment appends one Korean guidance block selected by the known question corpus: criminal, civil, administrative or IP. Preserve grade definitions, output schema, quote retry, model Qwen3:8b, thinking, temperature, truncation and generation limits. Both versions judge the same 12 labeled pairs per corpus; labels are copied and frozen before calls. Selection is seeded and stratified by available reference grade. Administrative references have ten negatives and only one example of each positive grade; report this limitation.

Use existing crossdomain-60-fanout references for three corpora and the latest IP calibration for IP. This is a retrospective diagnostic pilot, with existing AI references and old chunking for the three non-IP corpora. It is not a fresh holdout, structural-chunk comparison, legal correctness evaluation or retrieval precision estimate. Known corpus routing is an oracle; corpus labels may include questions outside the apparent substantive domain. Mixed-domain routing remains future work.

Freeze the current judge source and the four guidance blocks in a separate experiment directory. Run variants through the frozen module, replacing only its SYSTEM string; its existing hash includes that string and separates judgment keys. Source files, original labels, indexes, generation prompts and shared production default remain untouched. Reuse matching existing IP control judgments only with matching frozen content and model digest.

Report exact agreement, broad-relevance and direct-basis AC1/accuracy, missing judgments, quote retries, paired wins/losses and changed-grade source review per corpus. Preselect exact agreement as primary descriptive outcome. Promote no prompt automatically: a candidate must improve exact agreement, avoid broad-relevance regressions and receive source review. Small, uneven reference strata cannot establish a general improvement.

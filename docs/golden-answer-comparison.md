# Golden-answer comparison

This experiment compares original and smaller structural chunks using the same
80 questions from the historical answer evaluation. Both arms search four indexes
without domain filters or neighboring passages. It measures chunking's effect on
answers; the corpus relevance-grader profiles do not change answer generation.

The sample contains 20 administrative, 20 civil, 20 criminal and 20 cross-domain
questions. It has no dedicated IP group. Both arms receive fresh Gemini answers
with the same shared answer prompt, settings and 7500-character evidence ceiling.
The reference answers are withheld from generation. Complete answers are graded
with the same v2 Qwen3:8B grader, no thinking and a 16384-token context.

## Start a new run

From the repository root, with Gemini selected in `.env`, Ollama running and all
eight original/structural Qdrant indexes available:

```bash
PYTHONPATH=src .venv/bin/python scripts/eval_golden_answers.py all \
  --out data/eval/reports/chat_pipeline/structural-golden-new-run
```

Use the same `--out` for each phase or to resume that new run. The commands below
address the completed historical run at the default path; its saved answers can
still be graded and reported, but its frozen sources differ from the current code:

```bash
PYTHONPATH=src .venv/bin/python scripts/eval_golden_answers.py judge
PYTHONPATH=src .venv/bin/python scripts/eval_golden_answers.py report
```

Completed rows are reused when their manifests match. Failed generation calls are
retried on resume. Failed grades remain unresolved; rerun the judge phase before
reporting if necessary. Generation stops after three consecutive API failures.
The two arms alternate generation order; requests are spaced at least 15 seconds
apart. Avoid editing the frozen generation sources during a run.

Results are saved under
`data/eval/reports/chat_pipeline/structural-golden-80-2026-10-03/`.
The paired report checks grade input fingerprints, source quotes and matching
settings. `comparison.md` reports conclusion agreement, reference agreement and
grounded agreement; `by-group.json` reports the four sample groups. Grounded
reference agreement is the primary metric. `case-review.md` includes answers and
grades for review; the complete source bundles remain in each arm's contexts file.

## Interpretation

This is a development comparison using existing AI references. Chunk size and
section boundaries change together, and actual evidence lengths may differ even
though the ceiling matches. Quote presence does not establish entailment. Human
calibration is still required, and these scores do not establish legal correctness.
Use the fresh paired original arm as the baseline; older runs had different evidence
and text truncation policies.

## Completed result (2026-10-04)

All 160 answers and grades resolved. Two quote/schema validation failures were
recovered by rerunning the judge phase; completed grades were reused. A report
error naming an unresolved judgment means the comparison is incomplete. Recover
without regenerating answers:

```bash
PYTHONPATH=src .venv/bin/python scripts/eval_golden_answers.py judge
PYTHONPATH=src .venv/bin/python scripts/eval_golden_answers.py report
```

| Automated metric | Original | Structural |
| --- | ---: | ---: |
| Conclusion agreement | 39/80 (48.75%) | 57/80 (71.25%) |
| Reference agreement | 39/80 (48.75%) | 57/80 (71.25%) |
| Grounded reference agreement | 38/80 (47.5%) | 56/80 (70.0%) |

Primary gain: +22.5 percentage points; 23 improved pairs and 5 regressed pairs.
The paired bootstrap 95% interval is [+11.25, +33.75] percentage points (10000
resamples, seed42). These are exploratory AI-reference results, not independently
held-out or human-calibrated legal correctness. The next step is reviewing changed
cases and checking the grader's reasoning against the saved source evidence.

## Maintained reporting code

The `report` phase now calls `legal_rag.evaluation.answer_reporting` directly.
Saved pairs, evidence, manifest fingerprints and retry-policy provenance are
validated by `answer_run`. Report generation no longer executes Python stored
inside a historical experiment directory or modifies source text with replacements.
The existing report commands and output filenames remain the same; unavailable
support is reported with an explicit applicable-pair denominator.

The cleanup preserves the completed comparison JSON values, including deterministic
bootstrap intervals and group scores. Changing the runner source still triggers
its frozen generation-source check: reporting completed runs works, while resuming
generation with changed sources requires a separate run with fresh provenance.

## Maintained generation code (2026-10-05)

The `generate` phase uses `legal_rag.evaluation.answer_replay.AnswerReplay` with
explicit settings and a request limiter shared across both arms. It no longer
loads the historical replay with `runpy` or changes that script's globals. Four
HTTP attempts, quota/backoff handling, verification and regeneration policy,
usage records, and full-answer scoring are preserved. The historical `replay.py`
remains unchanged for provenance.

New selections freeze the maintained helper's hash. Existing selection manifests
are not rewritten: use a new output directory to generate with the current code.
The historical output path remains the default for saved-run judging/reporting.

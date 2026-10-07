# Answer grader v3 calibration

## Changes

V3 requires literal answer spans for each reference conclusion/condition and named legal basis. Validators reject invented spans, inconsistent checklist/summary labels, omitted reference article identifiers, and matched article identifiers absent from the answer. The rubric explicitly separates coverage, basis and evidence support; it includes the extra-detail and party-attribution failures found in the 28-case review.

The checklists are model-generated. A structurally valid checklist can still miss a condition, use the wrong legal regime, or misjudge entailment. V3 now uses constrained passage IDs for every quote and checklist item. Code copies literal text and derives coverage/contradiction/basis summaries from the selected statuses. Invalid semantic attribution (such as a claimed article match without that article in the answer) still triggers validation feedback and remains unresolved if repair fails. The current policy is `answer-checklist-ids-v4`; validated grades from earlier policies retain their recorded provenance. Its JSON schema requires A_NONE for missing checklist items and an actual answer passage for other statuses. Basis match choices are restricted per reference passage to answer passages containing all its article/case identifiers. Citation-presence information shows required identifiers, those missing from the full answer, and eligible answer passages. Retry feedback includes the previous selection and these constraints. Identifier presence still does not establish law identity or entailment; named laws without article/case numbers require semantic judgment.

V3 writes `answer_judgments-v3.jsonl`, `answer_judge-v3.json` and `answer_judge_attempts-v3.jsonl` in each arm directory of a separate calibration run. V2 defaults and historical scores are preserved. Existing evidence strings are graded as saved, without adding newly inferred roles.

## Regrade the saved 160 answers

Ollama must be serving `qwen3:8b`. Run from the repository:

```bash
uv run python scripts/calibrate_golden_answers.py all
```

The command processes both arms sequentially and resumes valid completed v3 grades. Rerun the same command after an interruption or unresolved grades. It uses a 16,384-token context and an 8,192-token output ceiling. The original 32,768-token default caused Ollama model-load HTTP 500 errors on this machine because insufficient RAM was available. The new output ceiling differs from v2 and is recorded in manifests. This compares grader configurations as a whole, not solely the prompt. Both arms use the same settings.

For only the paired report after grading:

```bash
uv run python scripts/calibrate_golden_answers.py report
```

Outputs: `data/eval/reports/chat_pipeline/structural-golden-80-2026-10-03-calibration-v3-ctx16384/comparison-v3.md` and `.json`. Frozen inputs and historical v2 judgments are copied with SHA256 checks; failed 32k artifacts remain in the original run. Use `--out` to select another sibling output directory. The JSON includes v2/v3 metric transitions and both judgments for changed records. An unresolved or stale record prevents a completed comparison. No Gemini calls, answer regeneration or reindexing occur in this command.

For blind human calibration of one arm:

```bash
uv run python scripts/judge_answers.py review \
  --run data/eval/reports/chat_pipeline/structural-golden-80-2026-10-03-calibration-v3-ctx16384/original \
  --out /tmp/original-answer-human-review.jsonl
```

Fill the exported manual fields independently, then compare against v3:

```bash
uv run python scripts/judge_answers.py report --rubric v3 \
  --run data/eval/reports/chat_pipeline/structural-golden-80-2026-10-03-calibration-v3-ctx16384/original \
  --manual /tmp/original-answer-human-review.jsonl
```

## Section roles for future chunks

New chunks and case evidence headers carry a section role derived only from explicit headings. Party claims, findings, facts, legal rules and orders keep separate roles; packing/merging cannot cross role boundaries. Unclassified or flattened text remains `unknown`, with an explicit warning in the rendered header. The shared generator prompt requires distinguishing allegations from adopted findings.

Some imported AIHub documents have already lost their headings, including the faculty qualification and unit-contract examples. This change cannot reconstruct their speaker boundaries. Recovering those boundaries requires the original structured fields; classifying them by the content of their prose would fabricate provenance.

Existing index chunks are not rewritten. To materialize the role changes, create a new corpus version with the same size settings used for the small-chunk experiment, for example:

```bash
uv run python scripts/rechunk_corpus.py \
  --corpus-version civil-v1 --output-version civil-struct800-roles-v2 \
  --target-chars 800 --max-chars 1000 --merge-threshold 400
```

Use the equivalent domain names for the other corpora. Inspect `chunks_by_role` in the new corpus report before indexing. Reindexing is needed for role text to affect dense/sparse retrieval and reranking. Saved-answer v3 calibration does not require it.

## Implementation status

The saved calibration now has 80 valid grades per arm. V3 conclusion/reference/grounded counts are 58/28/28 for original and 64/43/43 for structural. Accepted grades retain several repair-policy versions; this is not a uniform final-policy experiment. Human calibration and the impact of newly indexed role labels remain to be measured.

Server HTTP 5xx failures now stop the run after saving the response body, rather than retrying the same failure across every pair. Resolve the server error and rerun with the same settings to resume.

## Quote/checklist error recovery

Stop any already-running calibration process with Ctrl+C, then rerun the same `all` command. Valid existing grades are skipped; failed and pending pairs use the new ID policy. Do not delete the original judgments or manifests. This repair does not alter semantic checklist statuses to make grades pass.

## Repeated checklist timeout recovery

A streamed diagnostic of criminal case `a9489139e9877ca3` showed repeated identical R0/A0/covered items rather than a closed coverage array. Ollama remained responsive to a short request. Policy v4 bounds both checklists to `max(8, 2 * number of reference passages)`, requests unique items, and rejects duplicate reference/answer/status combinations. The limit leaves room for multiple conditions per passage, but it is an operational bound, not proof of exhaustive coverage. Valid earlier grades retain their policy provenance. Resume with the same `all --num-ctx 16384` command; no artifact deletion or server restart is required by this change. The completed saved run resolves all cases; its original arm retains historical policy versions, including two grades from policy v4.

## Codebase cleanup (2026-10-04)

- `answer_run` owns saved evidence precedence, provenance checks and validated grade loading. Missing evidence remains distinct from explicitly empty evidence.
- `answer_reporting` owns paired and v2-to-v3 reports. The scripts keep their existing commands and delegate reporting to this module.
- `answer_rubrics` binds the existing v2/v3 modules explicitly. `--rubric v2` remains the default; v3 requires explicit selection.
- `answer_judge_runner` owns the shared model-call lifecycle, manifest construction, resume decisions, diagnostics and retry/stop behavior. The legacy CLI path remains separate.

This refactor preserves rubric prompts, Pydantic schema names, citation/checklist
rules, manifest values and policy versions. Saved original report artifacts were
left untouched. Reports regenerated from copies of the completed runs reproduce
the existing comparison JSON values. No live regrading, indexing or test suite
was run for this cleanup; live generation and operational error paths still need
separate verification if requested.

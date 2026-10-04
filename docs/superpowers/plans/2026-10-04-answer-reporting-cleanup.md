# Answer Reporting Cleanup Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task in order. Perform the second plan only after this plan's compatibility checkpoint. Do not delegate unless the user requests it. Implementation was authorized by the user with "execute them". Optional test and live-model checks require separate authorization.

**Goal:** Replace experiment-specific reporting code execution with reusable, validated saved-answer reporting while retaining existing commands and result formats.

**Architecture:** Introduce a saved-run reader in `legal_rag.evaluation.answer_run` and paired comparison functions in `legal_rag.evaluation.answer_reporting`. The golden-answer and calibration scripts become report adapters. Provider calls, generation and retrieval remain outside these modules.

**Tech Stack:** Python 3.11, pathlib, dataclasses, Pydantic, existing v2/v3 evaluation modules, JSON/JSONL and Markdown.

---

## Order and scope

This is **phase 1**. Next: [Grader cleanup](2026-10-04-answer-grader-cleanup.md).

The current reporting dependency is `scripts/eval_golden_answers.py:report()` reading `data/eval/reports/chat_pipeline/structural-answer-pair-2026-10-03/summarize.py`, applying string replacements, then executing it. `scripts/calibrate_golden_answers.py` separately implements similar loading, validation, grouping and comparison logic and imports private helpers from `judge_answers.py`.

Only the reporting paths and their shared artifact helpers are in scope. Do not change dataset membership, references, prompts, chunking, model settings, generation replay, neighbor settings or rubric semantics. The generation path's existing `runpy` replay dependency is a separate future cleanup.

## Files and responsibilities

| File | Action | Responsibility |
|---|---|---|
| `src/legal_rag/evaluation/answer_run.py` | Create | Artifact reads, evidence precedence, hashes, provenance and grade validity |
| `src/legal_rag/evaluation/answer_reporting.py` | Create | Paired comparisons, groups, bootstrap and report rendering |
| `scripts/eval_golden_answers.py` | Modify | Thin v2 paired-report adapter |
| `scripts/calibrate_golden_answers.py` | Modify | Thin v3 comparison and v2-transition report adapter |
| `scripts/judge_answers.py` | Modify narrowly | Delegate shared artifact helpers through existing private wrapper names |
| `.gitignore` | Modify narrowly | Allow the two maintained experiment entry points to be tracked |
| `docs/golden-answer-comparison.md` | Update | Explain maintained reporting entry point |
| `docs/answer-grader-v3-calibration.md` | Update | Explain shared reader and compatibility limits |

## Task 1: Establish the compatibility baseline

- [x] Record `git status --short` and preserve unrelated working-tree changes.
- [x] Inspect current output keys in `comparison.json`, `by-group.json`, `comparison-v3.json`, and both Markdown reports. Record the filenames and existing key types before replacing any code.
- [x] Record SHA256 values for input pairs, verification/context files, judgments, base manifests and retry manifests in both completed experiment directories. Reports must not rewrite these artifacts.
- [x] Record these observed baselines from the completed 80-question run:

| Rubric | Arm | Conclusion | Reference agreement | Grounded |
|---|---|---:|---:|---:|
| v2 | original | 39 | 39 | 38 |
| v2 | structural | 57 | 57 | 56 |
| v3 | original | 58 | 28 | 28 |
| v3 | structural | 64 | 43 | 43 |

- [x] Preserve v3's 87 changed arm-level metric records and 18 structural grounded wins / 3 losses. Record policy distribution: original has 2 checklist-v1, 15 ids-v2, 61 ids-v3 and 2 ids-v4 grades; structural has 80 ids-v3 grades.

## Task 2: Extract the saved-run reader

- [x] Move `_jsonl`, `_evidence_by_key` and `_retry_manifest_hashes` into `answer_run.py` with public names `read_jsonl`, `evidence_by_key`, and `retry_manifest_hashes`. Preserve latest-record-per-key semantics and evidence precedence exactly: embedded pairs, then saved verify evidence overrides pairs, then context bundles fill missing keys. Distinguish missing evidence (`None`) from explicit empty evidence (`[]`).
- [x] Add `load_answer_run(directory: Path, rubric: str, *, strict: bool) -> AnswerRun`. `AnswerRun` carries `pairs`, `records`, `grades`, `scores`, `manifest`, `no_answer_keys` and `unresolved_keys`.
- [x] Implement one validation sequence: manifest hash; input fingerprint; recorded retry-policy hash; no-answer eligibility; rubric-specific Pydantic schema; quote/checklist validation; metric derivation. Recompute metrics from grades rather than trusting saved metric fields.
- [x] Support v2/v3 only in this reader. Keep legacy grading/reporting on its existing path. Initially choose existing modules explicitly; phase 2 replaces that selection with the rubric registry.
- [x] In strict mode, raise an actionable error containing directory, rubric, case key and cause for missing, stale, malformed or unresolved grades. In non-strict mode, classify them as unresolved so CLI reports retain their all-case bounds. Do not count unresolved grades as incorrect answers.
- [x] Retain the private CLI wrapper names for compatibility:

```python
def _jsonl(path: Path) -> list[dict]:
    from legal_rag.evaluation.answer_run import read_jsonl
    return read_jsonl(path)

def _evidence_by_key(run: Path) -> dict[str, list[str]]:
    from legal_rag.evaluation.answer_run import evidence_by_key
    return evidence_by_key(run)

def _retry_manifest_hashes(run: Path) -> dict[str, str]:
    from legal_rag.evaluation.answer_run import retry_manifest_hashes
    return retry_manifest_hashes(run)
```

Translate shared reader exceptions to `typer.BadParameter` only at CLI boundaries. Package modules must not import CLI scripts or Typer.

## Task 3: Extract paired comparison and rendering

- [x] Add `compare_answer_runs(original: AnswerRun, structural: AnswerRun) -> dict`. Validate identical keys, question/reference/dataset/gold fields, and judge manifests before calculating paired scores.
- [x] In the golden report adapter, preserve the existing generation-manifest comparison that ignores only `arm` and `contexts_sha256`. Calibration may lack copied generation manifests; do not invent them or newly require unavailable files for historical calibration.
- [x] Calculate sample size from loaded paired keys. Preserve the golden runner's 80-question input contract outside the reusable library. Eliminate hardcoded `/20` and replacement-based sample conversions.
- [x] For each metric, retain original/structural counts, improved/regressed case IDs, difference and deterministic bootstrap interval (10,000 samples; seed 42; existing percentile positions). When support is unavailable, calculate on applicable pairs and expose that denominator; never cast `None` to `False` silently.
- [x] Produce grouped comparisons through the same function, with existing dataset names. Preserve v2 `by-group.json` keys and v3 `groups` keys through explicit format adapters.
- [x] Add `compare_rubric_versions(old: AnswerRun, new: AnswerRun) -> dict`. Preserve v2-to-v3 transition labels and the complete old/new grades for changed metric records. Validate that the question/reference/answer/evidence inputs actually match between rubric runs.
- [x] Add `write_golden_report(run: Path) -> dict` and `write_calibration_report(run: Path) -> dict`. Retain golden filenames `comparison.json`, `comparison.md`, `case-review.md`, `by-group.json` and calibration filenames `comparison-v3.json`, `comparison-v3.md`. Preserve existing JSON keys; add explicit denominators only where necessary. Render contextual descriptions from named fields rather than replacing text from historical scripts.
- [x] Build all report data and validate both arms before writing reports. Each output is replaced atomically to avoid partially written files after interruption. Historical scripts under `data/` remain untouched.

## Task 4: Replace entry-point reporting glue

- [x] Replace the golden script's report function with:

```python
def report(out):
    from legal_rag.evaluation.answer_reporting import write_golden_report
    write_golden_report(out)
```

- [x] Replace calibration reporting with:

```python
def report(run):
    from legal_rag.evaluation.answer_reporting import write_calibration_report
    write_calibration_report(run)
```

- [x] Remove calibration's duplicate `load_arm` and imports of private CLI helpers once all callers use the shared reader. Retain preparation, judging arguments, defaults and frozen-input copying.
- [x] Remove only imports made unused by the reporting extraction. Preserve imports still required by generation or preparation.
- [x] Replace `.gitignore`'s `scripts/` entry with this narrow pattern; retain other ignore rules:

```gitignore
scripts/*
!scripts/eval_golden_answers.py
!scripts/calibrate_golden_answers.py
```

Existing tracked scripts remain tracked. Do not broadly expose local scratch scripts. Check the status of newly visible files before any staging.
- [x] Update documentation to name the reusable modules and unchanged commands. Note that changing the runner's source hash intentionally prevents generation resume against an old frozen source manifest; historical **reporting** must remain compatible. Do not bypass that generation protection.

## Task 5: Compatibility checkpoint before phase 2

Planning does not run checks or grant permission to add tests. On implementation, perform source/syntax and diff inspection. If the user authorizes verification, use these read-only report commands against copies of completed runs so historical reports are not overwritten:

```bash
.venv/bin/python scripts/eval_golden_answers.py report --out /tmp/legal-rag-report-cleanup-v2
.venv/bin/python scripts/calibrate_golden_answers.py report --out /tmp/legal-rag-report-cleanup-v3 --num-ctx 16384
```

The copied directories must contain the existing completed arm artifacts; these commands make no model calls.

- [x] Confirm counts, paired transitions, group results and bootstrap intervals against Task 1. Explain any intentional presentation-only differences.
- [x] Confirm input/judgment/manifest hashes and source experiment files have not changed.
- [x] Confirm source/syntax validity and `git diff --check`; ensure no maintained report function reads executable Python from `data/`.
- [ ] If tests are explicitly requested, add offline coverage for 20/80-question samples, stale manifests, invalid retry hashes, literal quote failures, no-answer versus generation error, missing versus empty evidence, unavailable support and mismatched paired inputs. Use temporary fixtures and no live model endpoints.
- [x] Review the diff and hand off to phase 2 only after reporting compatibility is established. Do not stage, commit or discard unrelated changes automatically.

## Completion criteria

Both existing report commands work through maintained package modules. Reports no longer execute experiment scripts or perform source-text replacement. Completed results retain their scores and provenance. Shared artifact validation is reusable by the grader refactor. No retrieval, generation or evaluation-rule changes are included.

## Execution record

Implemented sequentially in `.worktrees/answer-cleanup` using copies of the current
working files. Original artifacts and pre-existing modifications were captured
before edits. Compatibility checks used saved-artifact copies under `/tmp`:
`comparison.json`, `by-group.json` and `comparison-v3.json` retain identical JSON
values. Base and current retry manifests rebuild identically for both rubrics.
Source/syntax and diff inspection completed. Optional tests and live model resume
checks were not run. Cleanup files are copied back without staging or committing.

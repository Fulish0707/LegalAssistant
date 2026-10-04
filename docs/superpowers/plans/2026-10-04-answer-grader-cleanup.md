# Answer Grader Cleanup Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. First complete the [reporting cleanup checkpoint](2026-10-04-answer-reporting-cleanup.md). Do not delegate unless the user requests it. Implementation was authorized by the user with "execute them". Optional test and live-model checks require separate authorization.

**Goal:** Make v2/v3 rubric selection explicit and share the saved-answer judging lifecycle while preserving grading behavior, commands and artifact compatibility.

**Architecture:** A small immutable rubric descriptor binds each existing evaluation module to its schema and selection functions. A versioned judge runner owns model calls, manifests, resume, diagnostics and retries. The CLI retains `pairs`, `judge`, `report`, `review` and the historical legacy path.

**Tech Stack:** Python 3.11, dataclasses, Pydantic, httpx, existing OllamaChatModel and Typer CLI.

---

## Order and scope

This is **phase 2**, after [reporting cleanup](2026-10-04-answer-reporting-cleanup.md). Phase 1's shared saved-run reader is a required dependency.

The current `_judge_v2` handles both v2/v3 by reassigning v2-named schema/constants/functions. `_report_v2` repeats rubric selection and grade loading. Refactor this orchestration without changing either rubric's accepted grades or generation constraints.

V3 remains explicitly selected with `--rubric v3`; v2 remains the default. This cleanup does not adopt v3 as a more accurate evaluator or modify missing/mismatch semantics.

## Files and responsibilities

| File | Action | Responsibility |
|---|---|---|
| `src/legal_rag/evaluation/answer_rubrics.py` | Create | Explicit v2/v3 module bindings and policy descriptors |
| `src/legal_rag/evaluation/answer_judge_runner.py` | Create | Versioned judge lifecycle and append-only diagnostic records |
| `src/legal_rag/evaluation/answer_run.py` | Modify | Use registry for schemas/validators; preserve phase 1 validation |
| `scripts/judge_answers.py` | Modify | CLI adapters, legacy path and existing public commands |
| `src/legal_rag/evaluation/answer_judge.py` | Preserve rules | Existing v2 schema, prompts, metrics and selection behavior |
| `src/legal_rag/evaluation/answer_judge_v3.py` | Preserve rules | Existing v3 checklists, IDs, citation constraints and bounds |
| `docs/answer-grader-v3-calibration.md` | Update | Explain shared runner and unchanged rubric selection |

## Task 1: Capture the grader contract

- [x] Record the working-tree baseline and verify phase 1 completed successfully.
- [x] Inventory every existing judge option and default: model `qwen3:8b`, URL `http://localhost:11434`, context 12,288, output ceiling 4,096, rubric v2; non-legacy thinking disabled unless explicitly requested. Calibration separately passes 16,384 context and 8,192 output.
- [x] Capture base and retry manifests for both rubrics and their canonical hashes. Preserve schema names, schema JSON, system prompts, versions, policy fields and hash serialization. Pydantic model renaming can change schema hashes; do not rename persisted schema classes.
- [x] Record the current base manifest content, retry manifest content and diagnostic field names directly from `_judge_v2`. Preserve them as the compatibility contract, including HTTP 5xx handling currently applied to both versioned rubrics. Do not silently align runner behavior with differently worded historical policy metadata.
- [x] Capture successful-resume behavior: all completed valid grades skip model calls, including historical accepted retry policies. Failed or stale grades are re-evaluated; no-answer eligibility remains unchanged.

## Task 2: Introduce explicit rubric bindings

- [x] Add `answer_rubrics.py` using the existing modules as the source of truth. Keep legacy separate:

```python
from dataclasses import dataclass
from types import ModuleType

from . import answer_judge as v2
from . import answer_judge_v3 as v3


@dataclass(frozen=True)
class AnswerRubric:
    name: str
    module: ModuleType
    initial_selection: bool

    @property
    def grade_schema(self):
        return self.module.AnswerGradeV3 if self.name == "v3" else self.module.AnswerGradeV2


RUBRICS = {
    "v2": AnswerRubric("v2", v2, False),
    "v3": AnswerRubric("v3", v3, True),
}


def get_answer_rubric(name: str) -> AnswerRubric:
    try:
        return RUBRICS[name]
    except KeyError:
        raise ValueError(f"Unsupported versioned answer rubric: {name}") from None
```

- [x] Access system text, version constants, metric calculation, materialization and validation through `rubric.module`; access schema through `rubric.grade_schema`. Do not rebind variables named `AnswerGradeV2` or `V2_SYSTEM` to v3.
- [x] Update `answer_run.load_answer_run` to call the registry. Ensure the registry depends only on rubric modules and introduces no cycle with the reader, runner or reporting module.
- [x] Move the existing retry-manifest builder into a named helper owned by the runner. Use a shared common dictionary and the same existing v3 supplement. Produce exactly the original dictionary values; do not add new fields under an existing policy version.

## Task 3: Extract the versioned judge lifecycle

- [x] Add immutable `JudgeOptions` in `answer_judge_runner.py` with `model`, `base_url`, `think`, `num_ctx`, and `max_tokens`.
- [x] Add `judge_saved_answers(run: Path, rubric: AnswerRubric, options: JudgeOptions, *, emit: Callable[[str], None]) -> None`. Port the versioned loop in small sections: artifact loading; model digest; manifest comparison; previous-grade acceptance; invocation construction; attempt generation; result append; stop conditions; client cleanup.
- [x] Reuse `answer_run` helpers for evidence, latest records and retry provenance. Keep the report reader and judging path on the same validity rules.
- [x] Use rubric-specific initial selection: v2 starts with literal quote grading and repairs known/new structured-output or quote failures via IDs; v3 starts with IDs. Preserve the historical-error logic that decides whether a v2 retry should begin with selection mode.
- [x] Preserve two attempts per invocation. Structured-output/validation failures may repair on attempt 2; timeouts end that case without immediate retry; two consecutive timeouts stop the run; HTTP 5xx saves the response body and stops. Preserve counters, transport-error behavior, temperatures, token budgets and response metadata.
- [x] For v3, call its existing `selection_feedback` only for failed selection attempts with a catalog. Preserve status-dependent passage choices, citation-presence data, dynamic checklist bounds and duplicate rejection unchanged.
- [x] Append and flush attempt diagnostics and final case records with their current keys, policy hashes, input hashes and invocation IDs. Accepted historical records must not be rewritten or relabeled by the cleanup.
- [x] Close the model client on every exit path. Keep resource cleanup in the runner; map operational stop conditions to a CLI exit in the adapter.
- [x] Use a package-level `JudgeStoppedError` carrying the saved operational message for controlled stops. The CLI prints that message and exits 1. Propagate unrelated exceptions without hiding their cause.

## Task 4: Simplify CLI and report orchestration

- [x] Replace `_judge_v2` with the neutral `_judge_versioned` adapter. Preserve the `judge` command's signature and default rubric, and keep legacy execution unchanged:

```python
def _judge_versioned(run, model, base_url, think, num_ctx, max_tokens, rubric):
    from legal_rag.evaluation.answer_judge_runner import (
        JudgeOptions, JudgeStoppedError, judge_saved_answers,
    )
    from legal_rag.evaluation.answer_rubrics import get_answer_rubric

    options = JudgeOptions(model, base_url, think, num_ctx, max_tokens)
    try:
        judge_saved_answers(run, get_answer_rubric(rubric), options, emit=typer.echo)
    except JudgeStoppedError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
```

- [x] Rename `_report_v2` to `_report_versioned` and feed it `load_answer_run(..., strict=False)`. Preserve group counts, all-case bounds, policy reporting, manual-review fingerprint checks and AC1 calculations. Keep CLI presentation separate from artifact validity.
- [x] Preserve private artifact helper wrappers introduced in phase 1 until a caller search confirms no consumers still require them. Do not remove helpers because their names begin with an underscore.
- [x] Keep `review` exports blind and `pairs` source matching unchanged. Remove only imports proven unused after the extraction.
- [x] Confirm `calibrate_golden_answers.py` still launches the same judge command and records identical configuration. Do not change its completed-run location or rewrite stored manifests.

## Task 5: Compatibility checkpoint

No tests or model runs are executed as part of planning. Source/syntax and diff inspection are the implementation baseline. If the user authorizes verification, run reporting against temporary copies of the same completed artifacts and compare phase 1 output numerically:

```bash
.venv/bin/python scripts/judge_answers.py report --rubric v2 --run /tmp/legal-rag-report-cleanup-v2/original
.venv/bin/python scripts/judge_answers.py report --rubric v3 --run /tmp/legal-rag-report-cleanup-v3/original
.venv/bin/python scripts/calibrate_golden_answers.py report --out /tmp/legal-rag-report-cleanup-v3 --num-ctx 16384
```

- [x] Confirm the same twelve metric totals across both arms and rubrics, domain summaries, transitions and mixed-policy provenance from phase 1.
- [x] Confirm no hashes in accepted base/retry manifests change solely because code moved. If any changes appear, resolve the accidental compatibility change before claiming completion.
- [ ] If tests are explicitly requested, use offline fake model responses and temporary artifacts to cover: both rubric choices; initial/repair selection behavior; skip valid grades without generation; invalid/stale grades; no-answer versus generation failure; literal validation errors; duplicate checklist feedback; timeout stop; HTTP 5xx stop/body capture; transport failures; client cleanup; manual-review fingerprint rejection. Do not require live Gemini/Ollama calls.
- [ ] Any live resume check requires explicit verification authorization and runs on a temporary copy. A valid completed copy may query the model digest but must make zero generation calls and append zero grade records.
- [x] Inspect `git diff --check` and ensure CLI help/defaults are unchanged. No product route renaming, prompt edits, reindexing, dataset filtering or report metric redesign belong in this diff.
- [x] Review all touched files before optional focused commits. Preserve unrelated work and report remaining limitations.

## Completion criteria

CLI entry points delegate versioned grading to one runner with explicit rubric selection. v2/v3 rule definitions stay separate and unchanged. Saved grades, schemas, manifests, policy histories and report scores remain compatible. Timeout and citation safeguards remain intact. The code supports continued first-step consultation improvements without treating exact citation agreement as a new product requirement.

## Execution record

Implemented sequentially in `.worktrees/answer-cleanup` using copies of the current
working files. Original artifacts and pre-existing modifications were captured
before edits. Compatibility checks used saved-artifact copies under `/tmp`:
`comparison.json`, `by-group.json` and `comparison-v3.json` retain identical JSON
values. Base and current retry manifests rebuild identically for both rubrics.
Source/syntax and diff inspection completed. Optional tests and live model resume
checks were not run. Cleanup files are copied back without staging or committing.

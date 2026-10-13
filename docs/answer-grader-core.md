# Answer grader: core conclusion (default since 2026-10-06)

`scripts/judge_answers.py judge --run <dir>` now grades with the `core` rubric
(`src/legal_rag/evaluation/answers/answer_judge_core.py`). `--rubric v2`, `v3` and `legacy` still run as before.

## What it asks

One question per answer: does the answer state the reference's conclusion?

1. The reference is reduced to one core conclusion and an optional condition, once per question. The result is
   kept in `data/eval/judge_cache/core_conclusions.json` (`--conclusions`), keyed by prompt, model, question and
   reference, so every run and arm is measured against the same sentence.
2. The answer is split into numbered sentences. The grader says whether the core is stated, which sentences say
   it, and whether the answer contradicts it. It copies nothing.
3. Code computes the verdict: `conclusion_correct` = stated in a sentence that exists, and not contradicted.
   The condition is recorded and not required.

Output: `answer_judgments-core.jsonl` and `answer_judge-core.json` in the run directory. Graded answers are
skipped on resume. About 2 to 5 seconds per answer on Qwen3:8B, no thinking, 8192 context.

## Why v2 was replaced as the default

The v2 grader had to copy quotes from the answer and the evidence. When the first attempt mis-copied one, the
runner graded again in "selection mode" under a different instruction. Over every stored run of 2026-10-04 to
10-06, grades from an accepted first attempt were grounded in 56 of 317 (18%) and grades from the repair path
in 213 of 236 (90%). A run's grounded score was close to its number of repaired grades plus eight. Two arms
with the same prompt on identical evidence differed on 26 of 78 questions.

Do not use v2 grounded scores as evidence about answer quality. That includes the structural-chunk
comparisons of 2026-10-03/04 and the Gemini comparisons of 2026-10-05.

## How it was chosen

Seven rounds on answers from two identical arms, each round recording what moved the score:
`data/eval/reports/chat_pipeline/judge-improvement-2026-10-06/LOG.md`.

| | identical arms differ on | agrees with one rater's labels (76 answers) | same input twice |
|---|---|---|---|
| v2 stored grades | 26 of 78 questions | 45 | depends on the path taken |
| core, strict wording | 11 of 80 | 62 | 80 of 80 |
| core, lenient wording (this default) | 6 of 80 | 62 | not rerun; the CLI reproduced the experiment on 80 of 80 |

The lenient wording passes an answer whose conclusion points the same way even if only part of the reasons or
listed items is given. It was chosen by the project owner on 2026-10-06; it did better on the unselected control
sample (26 of 28 against 24 of 28) and passes more answers that only make a general statement (9 against 3
wrongly passed on the issue sample).

## Limits

- Agreement with an AI-written reference, not legal correctness.
- It does not check that the evidence supports the answer. Citation integrity is the verifier's job
  (`verification/verifier.py`), not this grader's.
- The labels it was tuned against come from one rater (Claude) reading each answer's summary line; three were
  revised after seeing variant outputs.
- Compare arms on frozen evidence (`eval_chat_pipeline.py freeze`, `verify --frozen`) and report
  `eval_chat_pipeline.py overlap` beside the grade.

# Retrieval failure diagnostics

This workflow records why an exact source document was missed. It does not grade legal sufficiency, classify failures, or change the official retrieval evaluator. No embedding, chunking, indexing, BM25, fusion-weight, or reranker changes are made.

```bash
uv run python scripts/diagnose_retrieval.py \
  --corpus-version ip-v1 \
  --domain intellectual_property \
  --dataset data/eval/ip-pilot-200.jsonl \
  --retrieval-mode hybrid --limit 50 \
  --top-k 20 --deep-k 100 --seed 42 --only-failures
```

Use `--profile`, `--root`, and `--corpus-version` for the existing index configuration. This checkout's full IP corpus is `ip-v1`, not the default `v1`. The local profile searches its configured corpus fanout; the report records all searched corpora and applies the requested legal-domain filter to every leg. There is no gold document-type filter. Candidate depths are raised to at least `deep_k` on a diagnostic settings copy. Existing fusion weights and the configured reranker are retained and recorded.

QA defaults to `<processed_corpus>/qa.jsonl`. Imported corpora can resolve the processed source's sibling `qa.jsonl` through their manifest; use `--qa-file` if that path is unavailable or ambiguous. Answers are only report content, never retrieval input.

## Selection and reproducibility

`--dataset` is a retrieval evaluation JSONL (`query`, `relevant_document_ids`). Otherwise `<eval_dir>/<corpus_version>.jsonl` is used if present. If QA has explicit `split` labels, only evaluation/test/validation/dev rows are eligible, even with a supplied dataset. Unsplit QA is eligible only when no explicit splits exist. When no evaluation dataset or split exists, the full processed domain QA is used and identified as `unsplit_processed_qa` in the summary.

All target rows are evaluated before sampling. **`--limit 50` limits exported review cases, not retrieval queries.** Use an existing pilot evaluation dataset to bound an expensive live run. `--only-failures` selects only Top-K exact-gold misses. A seeded reservoir over stable case-ID order bounds retained retrieval evidence to the review limit. Identical source files, seed, and retrieval results reproduce selection. Source hashes and evaluated/selected case IDs are recorded. Case IDs include the QA source line so repeated QA rows remain distinct.

Duplicate-question lookup covers all QA rows in the requested domain, including rows outside the evaluated subset. It strips surrounding whitespace and collapses repeated whitespace only. `duplicate_question_count` counts QA rows; `candidate_gold_document_ids` contains distinct sorted IDs, including the current gold. Multi-positive recall is a diagnostic check against those IDs, not a replacement official score.

## Output and review

By default, each invocation creates a new directory at:

```text
reports/retrieval_diagnostics/<version>/<domain>/<mode>-<UTC timestamp>/
```

Use `--output` to choose a new directory. Existing directories are refused before retrieval, so neither raw evidence nor human annotations can be overwritten.

- `failures.jsonl`: selected cases with all returned candidates up to `deep_k`, actual available scores/traces, and gold chunk previews. Includes successes when `--only-failures` is absent.
- `missing_gold_corpus.jsonl`: target QA rows whose gold has no processed chunks. These are excluded from retrieval metrics and never searched.
- `review.csv`: editable annotations plus question, answer, gold evidence, and the first three retrieved previews. Full deep-candidate evidence remains in JSONL.
- `summary.json`: metrics over **all evaluated target rows**, before failure sampling; counts, percentages, configuration, input provenance and IDs.
- `evaluated.jsonl`: compact per-case outcomes, including successes, for paired comparisons.
- `target.jsonl`: reusable evaluation membership for comparison runs.
- `README.md`: category definitions and review instructions.

Fill `manual_category` (A–E), `manual_relevant_result_ranks` (JSON list), and `manual_notes` in `review.csv`. Join annotations to immutable evidence using `case_id`.

A: true retrieval failure. B: evaluation label ambiguity (another source is sufficient). C: source/preprocessing problem (expected evidence absent). D: ranking problem (useful deep candidate outside Top-K). E: chunking problem (evidence diluted or poorly split). These categories require human inspection; finding a gold at rank 42 does not automatically assign D or establish legal relevance.

`--preview-chars` defaults to 1200 and `--max-gold-chunks` to 10. Gold chunks are chosen by chunk order; the full count is retained. Missing evidence in a truncated preview is not proof of missing source content: inspect the original processed chunks when necessary.

## Metric interpretation and mode comparison

`gold_rank` is the first returned **chunk rank** with the exact gold document ID; repeated chunks from one document occupy separate ranks. This follows the diagnostic contract. The existing official evaluator deduplicates document IDs before ranking and remains unchanged; its numbers therefore need not equal diagnostic recalls.

The summary reports exact Recall@5/10/20/50/100 only where the search depth supports them, plus requested Top-K/deep-K recall, MRR truncated at `deep_k`, exact-question multi-positive recall, and three rank buckets. Undefined rates for an empty evaluable set are JSON null. Missing-corpus cases are not ordinary retrieval failures. An exact miss is not proof that no useful legal evidence was retrieved.

Run dense and sparse using the hybrid run's `target.jsonl` (or the same original dataset):

```bash
uv run python scripts/diagnose_retrieval.py \
  --corpus-version ip-v1 --domain intellectual_property \
  --dataset data/eval/ip-pilot-200.jsonl \
  --retrieval-mode dense --limit 50 --top-k 20 --deep-k 100 --seed 42 --only-failures

uv run python scripts/diagnose_retrieval.py \
  --corpus-version ip-v1 --domain intellectual_property \
  --dataset data/eval/ip-pilot-200.jsonl \
  --retrieval-mode sparse --limit 50 --top-k 20 --deep-k 100 --seed 42 --only-failures
```

The complete evaluated target is identical across modes. Failure review subsets can differ because a success in one mode may fail in another. Compare `evaluated.jsonl` by case ID. In a multi-corpus configuration, dense-only/sparse-only still use the existing factory's fusion across same-kind corpus legs. No missing scores are fabricated; raw traces preserve available per-leg scores.

Many golds outside Top-K but inside deep-K support investigating ranking. Golds absent from deep-K cannot be recovered by reranking that candidate set. Neither observation settles label ambiguity or evidence sufficiency without manual review. No tuning should precede that review.

## One comparison report across modes

After the per-mode runs finish, combine them without retrieving again:

```bash
D=data/eval/reports/retrieval_diagnostics/ip-v1/intellectual_property
uv run python scripts/compare_retrieval_modes.py \
  $D/dense-seed42-final $D/sparse-seed42-final $D/hybrid-seed42-final \
  --output $D/comparison-seed42-final
```

The runs must share an identical `target.jsonl`, settings hash, Top-K/deep-K, domain and corpus version, and each must be a different mode; otherwise the script refuses. The output directory holds `comparison.md` (Recall@5/10/20/50/100, MRR@deep_k, Top-K / K+1–deep_k / not-found buckets, and per-case gold rank for each mode), `comparison.json` with the same content, and `cases.csv` with one rank column per mode (blank = not found). Cases whose gold is missing from the corpus are excluded from every mode's denominator.

## Graded relevance evaluation

Exact-gold metrics cannot measure this system. QA questions state general legal rules, and none of the 76,142 intellectual-property questions names a case number or document title, so hundreds of decisions can answer a question the gold document happens to answer. The graded evaluation asks instead whether retrieved chunks are relevant, judged by a local LLM that is checked against human labels. Design: `docs/superpowers/specs/2026-09-17-graded-relevance-eval-design.md`.

Grades are defined by whether the chunk answers the question's legal issue. The reference answer is only a hint about the expected rule and may contain errors, such as a wrong article number: **2** the chunk itself states the rule, standard or requirement that answers the issue (any decision stating it qualifies, not only the gold); **1** the same issue or provision, but the rule is not stated fully enough to answer alone, or a related field where facts are applied without stating the rule; **0** a different issue, institution, procedure or right (보정 vs 정정, 저작권 vs 특허권, 출원 거절 vs 등록 후 무효·정정), or shared vocabulary only.

Before grading, the judge copies one sentence of the chunk into `evidence_quote`. The quote is verified in code: after collapsing whitespace and stripping surrounding quote marks and ellipses, it must be at least 15 characters and occur in the chunk as the judge saw it (first 3,500 characters). A grade 2 whose quote fails is stored as 1. Grades 0 and 1 are never changed, so `quote_verified: false` on them is expected, not an error. Each row keeps `details`: `model_grade` (the judge's own grade), `evidence_quote` and `quote_verified`; rows from the first prompt have no `details`.

`--think/--no-think` (default `--no-think`) on `judge_relevance.py`, `judge_agreement.py` and `eval_graded.py` selects the thinking mode. It is part of the judgment key, so pass the same flag to all three. Think off generates up to 512 tokens, think on up to 4,096 (Ollama counts thinking tokens against the limit); both use `num_ctx` 8192. Each script prints the key as `prompt <hash12> think on|off`.

### Layout

`data/eval/relevance/<name>/`:

- `pool.json`, `pool.jsonl` — provenance and one row per (case, chunk): each mode's top-20 union plus the gold document's chunks. Only the first 10 chunks (`--max-gold-chunks`) of a gold document are pooled.
- `runs/<mode>.jsonl` — each mode's ranking to depth 100.
- `calibration.csv`, `calibration.json` — 50 stratified pairs for human grading; strata stay in the JSON.
- `judgments.jsonl` — append-only, keyed by judge model and prompt hash (which covers prompt text, schema, truncation, temperature, token and context limits, thinking mode and the quote verification rule). Changing any of these starts a new key; nothing is overwritten.
- `judge_runs.jsonl` — one provenance line per completed judge run that passed the calibration check: judge key, Ollama model digest (best effort, not part of the key), `calibration.csv` hash, flags and counts.
- `agreement-<hash12>.json`, `calibration_review-<hash12>.csv` — judge versus reference labels, one pair per judge key (`<hash12>` is the prompt hash prefix). The JSON records the hashes of `calibration.csv`, `calibration.json` and `pool.jsonl`, the think mode, `reference_source`, `quote_downgrades` (calibration pairs whose grade 2 became 1 by quote verification) and `agreement_before_quote_check` (the same statistics on `model_grade`). The review CSV adds `evidence_quote`, `quote_verified` and `model_grade`. The unsuffixed `agreement.json`/`calibration_review.csv` come from the first prompt and are left alone.

### Steps

```bash
P=data/eval/relevance/ip-v1-pilot-200

# 1. Pool (minutes). --root keeps the settings hash identical to the diagnostic runs.
uv run python scripts/build_relevance_pool.py --profile local --root "$PWD" --corpus-version ip-v1 \
  --domain intellectual_property \
  --target data/eval/reports/retrieval_diagnostics/ip-v1/intellectual_property/hybrid-seed42-final/target.jsonl \
  --output $P

# 2. Calibration sample, then a human fills human_grade (0/1/2) in calibration.csv.
uv run python scripts/sample_calibration.py --pool-dir $P

# 3. Judge the same 50 and measure agreement (exit 1 unless all 50 have a judge grade and
#    binary grade-2 kappa >= 0.6; rerun with --retry-failed after parse failures).
uv run python scripts/judge_relevance.py --pool-dir $P --calibration-only
uv run python scripts/judge_agreement.py --pool-dir $P --reference-source human

# 4. Judge the whole pool (hours; interrupt and rerun freely).
uv run python scripts/judge_relevance.py --pool-dir $P

# 5. Graded comparison.
uv run python scripts/eval_graded.py --pool-dir $P --output $P/graded-qwen3-8b
```

`calibration.csv` leads with `case_number` and `search_query` (for example `특허법원 2021허2359`, `상표법 제35조`) so each source can be looked up quickly. A statute label ending in `앞` means the chunk sits just before that article. Save the file from Excel as "CSV UTF-8". `judge_relevance.py` refuses to run until every calibration pair has a human grade, so the human never sees judge grades first. If the `human_grade` column was filled by a model rather than a person, say so with `--reference-source "<model>"`: it is recorded in the agreement JSON and printed as `reference labels: <source>`, because agreement with AI labels is inter-model agreement, not human validation.

### Reading results

Lead with nDCG@10 (gain 2^grade − 1, ideal from every judged chunk for the case) and Success@k (a grade-2 chunk in the top k). Success@k and MRR (`mrr@20`) stop at the pool depth, since deeper ranks were never pooled; with a pool shallower than 20, Success@20 is not reported. Each is reported per chunk and per document. Pooled recall is not reported, because its denominator depends on the pool.

`min_judged_coverage@10` and `min_judged_coverage@20` must be 1.0 at chunk level for a comparison to be valid; `eval_graded.py --output` refuses otherwise. A new retrieval configuration, such as a reranker, surfaces chunks nobody judged. Extending an existing pool with a new configuration is not scripted yet. Until it is, build a new pool directory with `build_relevance_pool.py` (same target, including the new mode), copy `judgments.jsonl` into it, and run `judge_relevance.py` there: judgments are keyed by (model, prompt hash, case, chunk), not by pool directory, so pairs already judged are skipped and only the new ones are sent to the judge. `--pool-file` only selects which pairs to judge; it must come from the same target as the pool directory. Parse failures are stored as `grade: null`. They count as judged for coverage, so they never block a comparison, but take gain 0, stay out of the ideal ranking, and are reported as `failed@10` per case and `failed_top10_total` per mode. `judge_relevance.py --retry-failed` judges them again and appends a new row, which replaces the failure.

Scope: this pilot is ip-v1 with the `legal_domain=intellectual_property` filter, under which the other corpora's legs return nothing, so multi-corpus fan-out is not measured.

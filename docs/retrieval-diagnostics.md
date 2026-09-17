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

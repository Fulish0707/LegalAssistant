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

Exact-gold metrics cannot measure this system. QA questions state general legal rules, and none of the 76,142 intellectual-property questions names a case number or document title, so hundreds of decisions can answer a question the gold document happens to answer. The graded evaluation asks instead whether retrieved chunks are relevant, judged by a local LLM that is checked against reference labels (Codex, GPT 6 Astra). Design: `docs/superpowers/specs/2026-09-17-graded-relevance-eval-design.md`.

Grades are defined by whether the chunk answers the question's legal issue. The reference answer is only a hint about the expected rule and may contain errors, such as a wrong article number: **2** the chunk itself states the rule, standard or requirement that answers the issue (any decision stating it qualifies, not only the gold); **1** the same issue or provision, but the rule is not stated fully enough to answer alone, or a related field where facts are applied without stating the rule; **0** a different issue, institution, procedure or right (보정 vs 정정, 저작권 vs 특허권, 출원 거절 vs 등록 후 무효·정정), or shared vocabulary only.

Before grading, the judge copies one sentence of the chunk into `evidence_quote`. The quote is verified in code: after collapsing whitespace and stripping surrounding quote marks and ellipses, it must be at least 15 characters and occur in the chunk as the judge saw it (first 3,500 characters). A grade 2 whose quote fails is stored as 1. Grades 0 and 1 are never changed, so `quote_verified: false` on them is expected, not an error. Each row keeps `details`: `model_grade` (the judge's own grade), `evidence_quote` and `quote_verified`; rows from the first prompt have no `details`.

`--think/--no-think` (default `--no-think`) on `judge_relevance.py`, `judge_agreement.py` and `eval_graded.py` selects the thinking mode. It is part of the judgment key, so pass the same flag to all three. Think off generates up to 512 tokens, think on up to 4,096 (Ollama counts thinking tokens against the limit); both use `num_ctx` 8192. Each script prints the key as `prompt <hash12> think on|off`.

### Layout

`data/eval/relevance/<name>/`:

- `pool.json`, `pool.jsonl` — provenance and one row per (case, chunk): each mode's top-20 union plus the gold document's chunks. Only the first 10 chunks (`--max-gold-chunks`) of a gold document are pooled.
- `runs/<mode>.jsonl` — each mode's ranking to depth 100.
- `calibration.csv`, `calibration.json` — 50 stratified pairs for reference grading (`codex_grade`, `codex_note`); strata stay in the JSON.
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

# 2. Calibration sample, then the reference labeller fills codex_grade (0/1/2) and codex_note in calibration.csv.
uv run python scripts/sample_calibration.py --pool-dir $P

# 3. Judge the same 50 and measure agreement (exit 1 unless all 50 have a judge grade and
#    binary grade-2 kappa >= 0.6; rerun with --retry-failed after parse failures).
uv run python scripts/judge_relevance.py --pool-dir $P --calibration-only
uv run python scripts/judge_agreement.py --pool-dir $P --reference-source "GPT 6 Astra"

# 4. Judge the whole pool (hours; interrupt and rerun freely).
uv run python scripts/judge_relevance.py --pool-dir $P

# 5. Graded comparison.
uv run python scripts/eval_graded.py --pool-dir $P --output $P/graded-qwen3-8b
```

### Top-10 pool and grade ≥1 headline

Calibration v2 (qwen3:8b, think on, 50 pairs, reference labels by GPT 6 Astra) reached binary kappa 0.667 on relevant (grade ≥1) vs 0 but only 0.38 on grade 2 vs 0/1. The 8B judge is usable at the ≥1 boundary, not at grade 2, so headline metrics count grade ≥1 as relevant and graded metrics are reference only. Because that boundary was chosen after seeing the results, it is re-validated on a fresh sample (seed 43) before the full run. To halve judging time (think on is about 15 s per pair), the pool is cut to each mode's top-10 union:

```bash
T=data/eval/relevance/ip-v1-pilot-200-top10

# Derive without retrieving again: keeps pairs ranked <= 10 by any mode (gold chunks only if retrieved
# there, unless --include-unretrieved-gold), copies runs/, judgments.jsonl and judge_runs.jsonl.
uv run python scripts/derive_relevance_pool.py --from $P --output $T --pool-depth 10

# Fresh calibration sample that never repeats the first one; fill codex_grade, then judge it.
uv run python scripts/sample_calibration.py --pool-dir $T --seed 43 \
  --quota rank_1_3=25 --quota rank_4_10=25 --exclude-calibration $P
uv run python scripts/judge_relevance.py --pool-dir $T --calibration-only --think
uv run python scripts/judge_agreement.py --pool-dir $T --think --pass-boundary 1 --reference-source "<source>"

uv run python scripts/judge_relevance.py --pool-dir $T --think
uv run python scripts/eval_graded.py --pool-dir $T --think --threshold 1 --output $T/graded-qwen3-8b-think
```

`pool.json` of a derived pool records `derived_from` (source directory, its `pool.jsonl` hash and depth), `include_unretrieved_gold` and `judgments_copied_from`; deepening is refused. Without a `gold` quota, `sample_calibration.py` places gold chunks by rank like any other pair. Only gold chunks retrieved in the top 10 stay in the derived pool (10 of 461 for the pilot), so `gold_llm_grades` in its agreement JSON is close to empty and no longer checks the judge against the source. `judge_agreement.py` always reports `binary_relevant` (grade ≥1) and `binary_grade2`; `--pass-boundary 1|2` (default 2) picks which kappa must reach 0.6 and is recorded as `pass_boundary`. `eval_graded.py --threshold 1|2` (default 2) sets the grade that success@k, MRR, `precision@10` and `ndcg@10_binary` (gain 1, ideal from judged chunks at or above the threshold) count as relevant; it is recorded in `graded.json` and `cases.csv` and stated in the markdown header. At threshold 1, graded `ndcg@10` and `precision@10_grade1/2` move to a "Reference (graded)" table. The copied `judgments.jsonl` also holds grades for pairs outside the top-10 pool; `eval_graded.py` scores only judgments whose pair is in the directory's `pool.jsonl` (grades, documents and failures alike), so they never enter the nDCG ideal, and reports the ignored count as `out_of_pool_judgments` in `graded.json` and the markdown header. `failed parses` likewise counts in-pool failures only. At threshold 1, `cases_without_relevant` (graded nDCG) also moves to the Reference table.

`calibration.csv` leads with `case_number` and `search_query` (for example `특허법원 2021허2359`, `상표법 제35조`) so each source can be looked up quickly. A statute label ending in `앞` means the chunk sits just before that article. Save the file from Excel as "CSV UTF-8". `judge_relevance.py` refuses to run until every calibration pair has a `codex_grade`, so the labeller never sees judge grades first. The reference labels in this project come from Codex (GPT 6 Astra), not a person; name the source with `--reference-source` (default `codex`). It is recorded in the agreement JSON and printed as `reference labels: <source>`, because agreement with AI labels is inter-model agreement, not human validation. Calibration files written before the column rename use `human_grade`/`human_note`; they are still read, and a row that fills both names with different grades is refused.

### Fusion weight sweep

At grade ≥1 the equal-weight hybrid (chunk `ndcg@10_binary` 0.693, `precision@10` 0.657) scored below dense alone (0.785 / 0.753), because RRF gives the weak sparse leg the same vote as the strong dense leg. Whether a *weighted* hybrid beats dense can be answered without retrieving anything again. This pool applies `legal_domain=intellectual_property`, under which only the two ip-v1 legs return hits, and `deepen_retrieval` raised `k_leg` and `k_fused` to the run depth of 100. `runs/dense.jsonl` and `runs/sparse.jsonl` are therefore exactly the two ranked lists `RRFFuser` was handed, so re-fusing them under new weights reproduces what a re-run would have retrieved. Equal-weight re-fusion reproduces the recorded `runs/hybrid.jsonl` element for element at full depth 100 for all 200 cases, and a test asserts it against the real directory. **This shortcut is valid only under a single-corpus filter.** With multi-corpus fan-out, each recorded per-mode run is already fused across corpora, its per-leg ranks are lost, and it cannot be re-weighted this way.

```bash
W=data/eval/relevance/ip-v1-pilot-200-sweep

# Re-fuse; no retrieval, no judging. --sparse-weight is repeatable (default 0, 0.25, 0.5, 0.75, 1.0).
uv run python scripts/sweep_fusion_weights.py --pool-dir $T --output $W \
  --profile local --corpus-version ip-v1 --root . --think

uv run python scripts/judge_relevance.py --pool-dir $W --think          # only the pending pairs
uv run python scripts/eval_graded.py --pool-dir $W --think --threshold 1 --output $W/graded-qwen3-8b-think
```

The new directory holds the copied `dense.jsonl` and `sparse.jsonl` plus one `runs/hybrid_s<weight>.jsonl` per variant, which `eval_graded.py` reads as mode columns sorted by weight. Variant runs are written at the source `run_depth` (100), not at `--depth`: document-level metrics take a document's rank from its first chunk and never truncate the ranked list, so a run cut to 10 chunks would yield fewer than 10 distinct documents and score below the depth-100 legs by construction. `--depth` sets only the pool depth, which is what `ranks` and the pending-pair counts use. `pool.jsonl` is the source rows with `ranks` recomputed from the new runs, followed by a row per chunk a variant newly surfaced; chunk text for those comes from the processed corpus and a chunk missing there is an error, never a silently unjudgeable pair. `is_gold_chunk` for a new row comes from the case's `relevant_document_ids` in the `target.jsonl` recorded in `pool.json` (verified by its sha256), because a top-10 pool holds a gold row for only 7 of the 200 cases. `pool.json` records `fusion_sweep` (weights, `rrf_k`, variants) and `derived_from` (source directory and the sha256 of its `pool.jsonl` and leg runs). The script prints, per variant, how many top-10 pairs are already judged and how many are pending, then the distinct pending total — the extra judging cost. For the pilot that was 7 pairs (~2 minutes at 15.3 s/pair), because the copied `judgments.jsonl` already covers most of what re-weighting promotes.

`hybrid_s0.00` is byte-identical to `dense.jsonl` (a chunk only the zero-weighted sparse leg saw scores 0 and `RRFFuser` drops it), and `hybrid_s1.00` is byte-identical to the source `hybrid.jsonl`, so both are fully judged and serve as anchors: their metrics reproduce dense's 0.785 and hybrid's 0.693 at chunk level and dense's 0.789 at document level. Comparing those two files with `cmp` is the check that the sweep is sound. **The intermediate variants must be judged before they are compared.** Until then their `min_judged_coverage@10` is below 1.0, `eval_graded.py` prints the coverage warning and refuses `--output`, and the unjudged chunks count as grade 0, which understates exactly the variants under test.

### Cross-encoder reranking

The reranker is the last stage of `HybridRetriever`: it scores every (query, chunk text) pair with a cross-encoder and re-sorts the fused candidates, so it can only reorder what fusion already found. Install the optional extra and switch the backend:

```bash
uv sync --extra rerank      # torch + sentence-transformers, several GB
```

```yaml
rerank:
  backend: cross_encoder    # noop | cross_encoder
  model: BAAI/bge-reranker-v2-m3
  device: cuda              # cpu works, roughly an order of magnitude slower
```

`CrossEncoderReranker` loads the model on first use, never in the constructor, so importing it or building a retriever stays cheap and offline; tests inject a fake loader. Scores are the model's raw outputs, kept as they are: none are dropped, ties keep the fused order, and the returned hits carry `retriever='reranked'` with the rerank score, which the diagnostics writer records as `rerank_score`. Raise `retrieval.k_fused` when enabling it — the reranker only sees the fused top `k_fused`, and reranking 5 candidates buys nothing. On an RTX 4060 Laptop, `BAAI/bge-reranker-v2-m3` loads in about 9 s and scores 100 candidates in about 6 s per query at `batch_size` 16 and `max_length` 512, using roughly 2.9 GB of VRAM.

Evaluating it needs a recorded run, and reranking is a function of the query and the candidates' text alone, so a pool's recorded fused run can be reranked without retrieving again — the same shortcut the fusion sweep uses, and the comparison stays paired because every run ranks the same candidates:

```bash
W=data/eval/relevance/ip-v1-pilot-200-sweep
R=data/eval/relevance/ip-v1-pilot-200-rerank

# Reranks runs/hybrid_s0.75.jsonl (top 100 per case) and records the new order to depth 100.
uv run python scripts/rerank_relevance_pool.py --pool-dir $W --output $R \
  --profile local --corpus-version ip-v1 --root . --device cuda --think

uv run python scripts/judge_relevance.py --pool-dir $R --think          # only the pending pairs
uv run python scripts/eval_graded.py --pool-dir $R --think --threshold 1 --output $R/graded-rerank
```

The new directory holds `runs/reranked.jsonl` beside the copied `dense.jsonl`, `sparse.jsonl` and `hybrid_s0.75.jsonl`, the source pool rows extended with the chunks the reranked run newly lifts into the top 10 (chunk text from the processed corpus; a chunk missing there is an error, never an unjudgeable pair), and the copied `judgments.jsonl`, `judge_runs.jsonl`, `calibration.csv` and `calibration.json`. `pool.json` records the source directory with the sha256 of its `pool.jsonl` and of every run read, plus the rerank model, device, batch size, max length and the settings hash. Like the sweep's variants, the reranked run is written at the source `run_depth` of 100, not at `--depth`, so document-level metrics are not truncated by construction.

A reranker reorders deeply, so it promotes far more unjudged chunks than re-weighting does: for the pilot, reranking the fused top 100 lifted 1,315 new pairs into a top 10 and left 1,313 distinct pairs pending (~5.6 h at 15.3 s/pair) against the sweep's 7, while the three baselines stayed fully judged. Reranking 200 cases x 100 candidates itself took 20.1 min on the 4060 (6.0 s per case). Pass the same judge flags here as when the pool was judged (`--think` for the pilot): the pending count is computed for that judge key, and the wrong one reports every pair as pending. **Judge them before reading any metric.** Until coverage is complete, `min_judged_coverage@10` is below 1.0, `eval_graded.py --output` refuses, and the unjudged chunks count as grade 0 — which understates exactly the run under test.

### Reading results

Lead with nDCG@10 (gain 2^grade − 1, ideal from every judged chunk for the case) and Success@k (a chunk at or above `--threshold`, grade 2 by default, in the top k). Success@k and MRR (`mrr@20`) stop at the pool depth, since deeper ranks were never pooled; with a pool shallower than 20, Success@20 is not reported. Each is reported per chunk and per document. Pooled recall is not reported, because its denominator depends on the pool.

`min_judged_coverage@10` and `min_judged_coverage@20` must be 1.0 at chunk level for a comparison to be valid; `eval_graded.py --output` refuses otherwise. A new retrieval configuration, such as a reranker, surfaces chunks nobody judged. Two such extensions are scripted, both reusing a pool's recorded runs: `sweep_fusion_weights.py`, for re-weighted RRF under a single-corpus filter, and `rerank_relevance_pool.py`, for a cross-encoder reranking a recorded fused run (see above). For anything that needs real retrieval, build a new pool directory with `build_relevance_pool.py` (same target, including the new mode), copy `judgments.jsonl` into it, and run `judge_relevance.py` there: judgments are keyed by (model, prompt hash, case, chunk), not by pool directory, so pairs already judged are skipped and only the new ones are sent to the judge. `--pool-file` only selects which pairs to judge; it must come from the same target as the pool directory. Parse failures are stored as `grade: null`. They count as judged for coverage, so they never block a comparison, but take gain 0, stay out of the ideal ranking, and are reported as `failed@10` per case and `failed_top10_total` per mode. `judge_relevance.py --retry-failed` judges them again and appends a new row, which replaces the failure.

Scope: this pilot is ip-v1 with the `legal_domain=intellectual_property` filter, under which the other corpora's legs return nothing, so multi-corpus fan-out is not measured.

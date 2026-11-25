# k_fused=50 offline replay of the Priority 1 capture

- **Date:** 2026-09-29
- **Result type:** Development-only paired comparison; no serving setting changed

## Question and method

[Priority 1](../live-flask-current-2026-09-29-rootvenv/priority1-report.md) compared the current configuration (`k_fused=100`, `leg_floor=3`, fp16 reranker) with a run recorded on 2026-09-19 (`k_fused=50`, no floor, fp32). Three settings differ, so the result could not say which one moved the ranking. This replay separates them without retrieving or reranking again.

Each captured case holds all eight legs at depth 100 and the fp16 cross-encoder score of every fused candidate. The replay re-fuses the legs at `k_fused=50` with the floor at 3 (weights and RRF constant as served), which gives the candidates a `k_fused=50` search would have sent to the reranker, and orders them by the recorded scores (score descending, ties by fused order, as `CrossEncoderReranker` does). The 50 candidates are always a subset of the 100. Comparing three runs on the same questions and judgments splits the change in two:

- **Floor step:** 2026-09-19 run to the replay (`k_fused=50`, floor 3). Reproduction check 2 below shows fp16 and index drift add almost nothing to it.
- **`k_fused` step:** the replay to the current run (`k_fused=100`), everything else equal.

## Checks on the replay

1. **Parity at `k_fused=100`.** Replaying the captured settings reproduces the captured fused list, reranked order and scores, and the Priority 1 run files in all 260 cases. Every case has a score for each of its 100 candidates, and every `k_fused=50` candidate set has exactly 50 chunks.
2. **Reproduction of an independent run.** Replaying floor 0 at `k_fused=50` from today's legs and fp16 scores gives the 2026-09-19 top 10 as an identical ordered list in 189/200 IP and 57/60 mixed-domain cases, and as an identical set in 199/200 and 60/60. The one differing IP case is a near-tie at ranks 10 and 11 (0.3908 against 0.3907). The 2026-09-19 run used fp32 and an earlier index, so fp16 and index drift explain almost none of the change.
3. **Pointwise scoring.** The replay assumes a chunk's score does not depend on its batch. On 15 sampled cases (10 IP, 5 mixed) the real fp16 reranker, built as serving builds it, reproduced the recorded scores exactly at `k_fused=100` and within 6.7e-4 at `k_fused=50` (different batches), with an identical top 10 in 15/15 cases. This covers 15 of 260 cases, not all of them.

## Ranking changes

| | 2026-09-19 vs replay | replay vs current | 2026-09-19 vs current |
| --- | ---: | ---: | ---: |
| IP, mean shared top-10 chunks | 7.13 | 5.95 | 5.01 |
| Mixed, mean shared top-10 chunks | 8.83 | 7.47 | 7.65 |

On IP, 4.05 of the current top 10 fall outside the `k_fused=50` candidate set (2.53 on the mixed set). The share of IP top-10 slots held by ip-v1 chunks rose from 27.6% (2026-09-19) to 42.7% (replay) to 60.9% (current). A chunk is attributed to the corpus of the leg that returned it in this capture; the chunk's own `corpus_version` gives the same figures, and only 249 of 187,870 chunk occurrences appear in the legs of more than one corpus. The fan-out section of `docs/retrieval-diagnostics.md` gives 31.8% for the 2026-09-19 run because it attributes documents by id against the processed corpora, so compare shares only within one method. The mixed set's corpus shares moved by at most 4 points per step.

## Graded results

The 290 top-10 pairs the replay added to the Priority 1 pools were judged with the Priority 1 judge (`ollama:qwen3:8b`, prompt hash `7a2b826ac9cb`, `think=true`). Every top-10 pair of the three runs has a valid grade and no parse failed; the official `eval_graded.py` reports in `grading/*/graded-threshold{1,2}/` show 100% coverage. Intervals reuse Priority 1's paired bootstrap (10,000 resamples, seed `20260928`, stratified by domain, one cluster per question). The two Priority 1 runs reproduce Priority 1's own metrics exactly; nDCG@10 differs because the pool now holds the replay's pairs, so compare nDCG only within this table.

### IP, 200 questions

| Metric | 2026-09-19 | Replay: `k_fused=50`, floor 3 | Current: `k_fused=100` | Floor step | `k_fused` step |
| --- | ---: | ---: | ---: | ---: | ---: |
| Precision@10, grade >= 1 | 0.690 | 0.718 | 0.791 | +0.029 [+0.012, +0.045] | +0.073 [+0.052, +0.094] |
| Precision@10, grade 2 | 0.449 | 0.490 | 0.545 | +0.041 [+0.022, +0.060] | +0.054 [+0.032, +0.077] |
| nDCG@10 (graded) | 0.629 | 0.669 | 0.732 | +0.040 [+0.020, +0.060] | +0.063 [+0.040, +0.086] |
| Success@5, grade >= 1 | 0.970 | 0.965 | 0.990 | -0.005 [-0.025, +0.010] | +0.025 [+0.005, +0.050] |
| Success@5, grade 2 | 0.900 | 0.910 | 0.925 | +0.010 [-0.020, +0.040] | +0.015 [-0.015, +0.050] |
| Success@10, grade 2 | 0.945 | 0.955 | 0.975 | +0.010 [0.000, +0.025] | +0.020 [+0.005, +0.040] |
| MRR@10, grade 2 | 0.690 | 0.714 | 0.710 | +0.024 [-0.010, +0.059] | -0.004 [-0.037, +0.028] |

### Mixed domains, 60 questions

| Metric | 2026-09-19 | Replay: `k_fused=50`, floor 3 | Current: `k_fused=100` | Floor step | `k_fused` step |
| --- | ---: | ---: | ---: | ---: | ---: |
| Precision@10, grade >= 1 | 0.580 | 0.580 | 0.592 | 0.000 [-0.015, +0.015] | +0.012 [-0.013, +0.037] |
| Precision@10, grade 2 | 0.402 | 0.403 | 0.407 | +0.002 [-0.015, +0.017] | +0.003 [-0.018, +0.025] |
| nDCG@10 (graded) | 0.798 | 0.804 | 0.795 | +0.006 [-0.013, +0.027] | -0.008 [-0.029, +0.013] |
| Success@5, grade 2 | 0.917 | 0.900 | 0.867 | -0.017 [-0.050, 0.000] | -0.033 [-0.083, 0.000] |
| Success@10, grade 2 | 0.933 | 0.933 | 0.933 | 0.000 | 0.000 |
| MRR@10, grade 2 | 0.758 | 0.764 | 0.744 | +0.006 [-0.001, +0.018] | -0.021 [-0.043, -0.005] |

Success@10 at grade >= 1 is 0.99, 0.99 and 1.00 on IP and 0.967 throughout on the mixed set. Steps may not sum to the total because of rounding.

## Where the change lands

Precision by rank band, grade >= 1 (`rank-bands.json`). Flask builds its evidence from the top-ranked chunks until it holds five distinct documents, so ranks 1 to 5 are the part most likely to reach the model.

| | 2026-09-19 | Replay | Current | Floor step | `k_fused` step |
| --- | ---: | ---: | ---: | ---: | ---: |
| IP, ranks 1-5 | 0.730 | 0.777 | 0.809 | +0.047 [+0.028, +0.066] | +0.032 [+0.012, +0.052] |
| IP, ranks 6-10 | 0.649 | 0.659 | 0.772 | +0.010 [-0.015, +0.035] | +0.113 [+0.076, +0.150] |
| Mixed, ranks 1-5 | 0.673 | 0.687 | 0.657 | +0.013 [+0.003, +0.027] | -0.030 [-0.053, -0.010] |
| Mixed, ranks 6-10 | 0.487 | 0.473 | 0.527 | -0.013 [-0.043, +0.013] | +0.053 [+0.010, +0.097] |

On IP the floor lifts the top five and `k_fused=100` lifts mostly ranks 6 to 10. On the mixed set the floor is neutral or slightly positive, while `k_fused=100` moves relevant chunks out of the top five (grade 2 in ranks 1-5: -0.030 [-0.057, -0.007]) and into ranks 6 to 10.

## What Flask shows the model

Each run's top 10 was passed through Flask's formatter (five documents, 1,200 characters each). The density rows come from `evidence-precision.json`, the quote rows from `replay-analysis.json`. For the current run the rebuilt evidence equals the captured evidence in all 260 cases. `_with_legal_evidence` removes the search tool from a turn whose pre-search returned text, and all 260 questions trigger the pre-search, so this is all the statute and case-law text the model has for them.

| | 2026-09-19 | Replay | Current | Floor step | `k_fused` step |
| --- | ---: | ---: | ---: | ---: | ---: |
| IP, shown documents holding a relevant chunk (of 5), grade >= 1 | 3.645 | 3.860 | 4.020 | +0.215 [+0.120, +0.310] | +0.160 [+0.060, +0.260] |
| IP, same, grade 2 | 2.490 | 2.720 | 2.810 | +0.230 [+0.110, +0.350] | +0.090 [-0.025, +0.205] |
| IP, verified grade-2 quote visible | 0.860 | 0.860 | 0.860 | 0.000 [-0.040, +0.040] | 0.000 [-0.045, +0.050] |
| Mixed, shown documents holding a relevant chunk (of 5), grade >= 1 | 3.233 | 3.283 | 3.167 | +0.050 [-0.017, +0.117] | -0.117 [-0.233, 0.000] |
| Mixed, same, grade 2 | 2.383 | 2.417 | 2.283 | +0.033 [-0.050, +0.133] | -0.133 [-0.267, 0.000] |
| Mixed, verified grade-2 quote visible | 0.800 | 0.800 | 0.783 | 0.000 [-0.050, +0.050] | -0.017 [-0.083, +0.033] |

The quote row asks whether at least one verified grade-2 quote is visible. That was already true for 86% of IP questions, so the measure cannot register a denser evidence set, and it did not change at any step. Density did change on IP: of the five documents shown, 3.65 held a relevant chunk before and 4.02 do now (+0.375 [+0.240, +0.515] in total, 57% from the floor and 43% from `k_fused`). A document counts only when a relevant chunk kept text inside the 1,200 characters. On the mixed set the total is -0.067 [-0.183, +0.033], with the `k_fused` step slightly negative.

## Reading the result

- **The floor** raises or leaves unchanged the precision at the top of the ranking in both sets (mixed grade-2 Success@5 lost one question), and costs nothing at a fixed `k_fused` (measured on 2026-09-27: 1.34 s against 1.46 s per retrieve).
- **`k_fused=100`** trades a gain on IP (ranks 6 to 10 and, to a smaller degree, the top five) for a small loss at the top of the mixed-domain ranking. The loss is small and the mixed set has 20 questions per domain, so it needs confirmation on more data. The 2026-09-27 note records about 0.7 s more per retrieve for `k_fused=100`; this replay measured no latency.
- **fp16 and index drift** add almost nothing (check 2), so the 2026-09-19 to current change is the floor and `k_fused`.
- Which `k_fused` to serve depends on the expected question mix and the latency budget. Nothing here changes the served configuration, and this evidence does not make either value a clear winner on what the model sees.

## Limitations

- The questions are reused development data, the labels come from a local model (`qwen3:8b`), and the judge's calibration check was skipped; no independent human labels exist. Results are provisional.
- The mixed set has 60 questions. The intervals are not adjusted for the many metrics examined; read the mixed-set differences as a watch item.
- The replay tests `k_fused` 50 against 100 at floor 3, and the floor 0 against 3 at `k_fused=50`. Replaying the floor at `k_fused=100` would need scores for candidates the capture never scored, so an interaction between the two settings is untested.
- Scores are assumed pointwise; the real reranker confirmed this on 15 of 260 cases.
- The evidence measures count graded chunks and one Qwen-selected quote per pair. They do not measure answer quality, and they are not Recall@k.

## Artifacts

All paths are relative to this directory; every experiment directory is written once.

- `build_replay.py`, `manifest.json`, `replay-diagnostics.json`: replay, parity checks, pools and judge queues.
- `check_reproduction.py`, `reproduction-check.json`: check 2.
- `check_pointwise.py`, `pointwise-check.json`: check 3.
- `analyze_replay.py`, `replay-analysis.json`: graded and model-visible comparison; it stops if it cannot reproduce Priority 1's own numbers.
- `rank_bands.py`, `rank-bands.json`; `evidence_precision.py`, `evidence-precision.json`: the two tables above.
- `post_judge.sh`: the order in which the last steps ran.
- `grading/{ip-200-fanout,crossdomain-60-fanout}/`: pools (`pool.jsonl`, `runs/`), judgments (Priority 1's, plus the 290 new pairs), judge logs and the official graded reports.

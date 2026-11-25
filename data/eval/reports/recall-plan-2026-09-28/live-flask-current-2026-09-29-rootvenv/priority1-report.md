# Priority 1 — Current Flask retrieval measurement

- **Date:** 2026-09-29
- **Result type:** Development-only paired comparison

## Scope and capture

The capture follows the Flask `/chat` search path: the original query goes to unfiltered `/v1/search`, which returns the reranked top 10; Flask then keeps at most five documents and at most 1,200 characters per document. It uses the existing 200-question IP target and 60-question mixed-domain target. Both sets have already been used in development.

The current resolved settings were `k_dense=100`, `k_sparse=100`, `k_fused=100`, `leg_floor=3`, sparse weight `0.75`, RRF `k=60`, and `BAAI/bge-reranker-v2-m3` on CUDA in FP16 with max length 512. The paired historical run is the 2026-09-19 four-corpus fan-out with `k_fused=50`, sparse weight `0.75` and the same reranker model at max length 512. It predates `leg_floor` and the reranker `dtype` option, both added on 2026-09-27 with defaults of no floor and fp32, so it ran without a floor and in fp32. The comparison therefore changes three settings at once (`k_fused` 50 to 100, `leg_floor` 0 to 3, fp32 to fp16) and does not isolate `k_fused` by itself. The [k_fused=50 offline replay](../replay-kfused50-2026-09-29/replay-report.md) separates them. Replaying floor 0 at `k_fused=50` from today's legs and recorded fp16 scores reproduces the 2026-09-19 top 10 (identical ordered lists in 189/200 IP and 57/60 mixed-domain cases), so fp16 and index drift explain almost none of the change; it comes from the floor and `k_fused`. On IP, grade >= 1 Precision@10 rose by 0.029 from the floor and 0.073 from `k_fused`. On the mixed set the floor was neutral and `k_fused=100` lowered top-five precision.

The 3,077 chunks shared by the historical judging pool and this capture have identical text. The earlier labels came from the same judge model, prompt hash (which covers the generation settings and the quote-verification rule) and `think` mode as the new ones, and the judge module has not changed since before the first earlier label.

The current capture had 260/260 requests without retrieval errors and matched the live Flask evidence helper's ordered document quotes in all 20/20 parity cases. Across 260 offline formatter replays, all 260 ordered quotes matched exactly. The replay substituted captured `/v1/search` hits for the HTTP response; it checks grouping and truncation, not live transport or dynamic source labels. No serving code or configuration changed.

## Graded top-10 results

The same `ollama:qwen3:8b` judge, prompt hash `7a2b826ac9cb93494241c98ccc829755d2af0ee37ebd8e3178cdbd798ecb596c`, and `think=true` profile graded both runs. Every compared top-10 pair was judged; each official report recorded 100% minimum coverage and zero parse failures. The 95% intervals below use 10,000 paired bootstrap resamples, seed `20260928`, stratified by domain and clustered by `group_id` (one group per question here).

| Dataset and metric | 2026-09-19: `k_fused=50`, no floor, fp32 | Current: `k_fused=100`, floor 3, fp16 | Paired change, current minus prior (95% CI) |
| --- | ---: | ---: | ---: |
| IP, Success@10, grade >= 1 | 0.990 | 1.000 | +0.010 [0.000, +0.025] |
| IP, Success@10, grade 2 | 0.945 | 0.975 | +0.030 [+0.010, +0.055] |
| IP, Precision@10, grade >= 1 | 0.690 | 0.791 | +0.101 [+0.074, +0.128] |
| IP, Precision@10, grade 2 | 0.449 | 0.545 | +0.096 [+0.067, +0.124] |
| IP, graded nDCG@10 | 0.641 | 0.748 | +0.107 [+0.077, +0.136] |
| Mixed domains, Success@10, grade >= 1 | 0.967 | 0.967 | 0.000 [0.000, 0.000] |
| Mixed domains, Success@10, grade 2 | 0.933 | 0.933 | 0.000 [0.000, 0.000] |
| Mixed domains, Precision@10, grade >= 1 | 0.580 | 0.592 | +0.012 [-0.013, +0.037] |
| Mixed domains, Precision@10, grade 2 | 0.402 | 0.407 | +0.005 [-0.015, +0.023] |
| Mixed domains, graded nDCG@10 | 0.804 | 0.802 | -0.003 [-0.028, +0.024] |

On the mixed-domain set, grade-2 Success@5 fell from 0.917 to 0.867 (-0.050; 95% CI [-0.117, 0.000]) and grade-2 MRR@10 fell by 0.015 (95% CI [-0.035, -0.001]); grade-2 Success@10 stayed at 0.933. The Success@5 point changes were -0.100 for civil (19/20 to 17/20), -0.050 for criminal (18/20 to 17/20), and 0 for administrative (18/20 in both runs). These are small development samples, with 20 questions per domain. The Success@5 change is three questions. The MRR@10 interval just excludes zero (upper bound -0.001) and is one of many intervals examined, so read both as a watch item, not as a regression. On IP, grade-2 Success@5 rose from 0.900 to 0.925 (+0.025; 95% CI [-0.015, +0.065]) and grade >= 1 Success@5 from 0.970 to 0.990 (+0.020; [+0.005, +0.040]).

The top-10 ranking changed substantially on IP: an average 5.01 of 10 chunks overlapped, with the same ordered top 10 in 3/200 cases. Mixed-domain overlap was 7.65/10, with the same ordered list in 11/60 cases. The newly surfaced top-10 slots were all included in the judged pool.

## Exact-source proxy and stage loss

This dataset does not contain an exhaustive list of every relevant legal source, so classic corpus-level Recall@k cannot be calculated. The following exact-source metric is only a hit rate against each question's recorded reference document; it is a weak proxy for semantic recall, particularly for IP questions.

| Dataset | 2026-09-19 reference Hit@10 | Current reference Hit@10 |
| --- | ---: | ---: |
| IP, 200 questions | 0.010 (2/200) | 0.015 (3/200) |
| Mixed domains, 60 questions | 0.533 (32/60) | 0.567 (34/60) |

At current `k_fused=100`, recorded reference coverage by domain fell through these stages:

| Domain | In candidate union | In fused top 100 | In reranked top 10 | In Flask evidence |
| --- | ---: | ---: | ---: | ---: |
| IP | 10.5% | 3.0% | 1.5% | 1.5% |
| Administrative | 95% | 90% | 85% | 75% |
| Civil | 65% | 50% | 15% | 15% |
| Criminal | 95% | 90% | 70% | 65% |

These are exact-reference results, not a complete recall measure. Their contrast with judged semantic Success@10 (IP 1.000 at grade >= 1; mixed domains 0.967) shows that the recorded source document often is not the only useful source.

## Flask evidence survival

A grade-2 pair has a Qwen-selected quote that the judge verified as a substring of its retrieved chunk. The table checks whether that quote appears in the actual five-document / 1,200-character prepared evidence. The judge's quote collapses runs of whitespace that the source text keeps (double spaces, newlines), so quote and evidence are compared with whitespace collapsed, the same rule the judge applies when it verifies a quote. An exact substring test misreads those quotes as lost; it finds 310 of 1,089 IP quotes and 112 of 244 mixed-domain quotes. The judge also strips surrounding quote marks and ellipses before it verifies; this comparison does not, so the 5 IP and 2 mixed-domain quotes that cannot be placed in their chunk may be counted as lost though visible. This is a strict measure of one quote per pair, not a human-validated support-span recall score.

| Dataset | Grade-2 top-10 chunks | Quote visible | Lost: document outside first five | Lost: absent from a selected document |
| --- | ---: | ---: | ---: | ---: |
| IP | 1,089 | 490 (45.0%) | 475 (43.6%) | 124 (11.4%) |
| Mixed domains | 244 | 120 (49.2%) | 90 (36.9%) | 34 (13.9%) |

A quote counts as visible when it appears in its own chunk's excerpt (457 IP, 116 mixed) or in text shown through another chunk or document (33 IP, 4 mixed). Of the quotes lost inside a selected document, 20 IP and 14 mixed-domain chunks contributed no text to the 1,200 characters, and 104 IP and 20 mixed-domain quotes lie past the cut.

At least one grade-2 quote is visible for 171/195 IP questions and 46/56 mixed-domain questions that had a grade-2 result in the top 10 (172 and 47 counting text shown through another chunk or document). At least some text from a grade >= 1 top-10 chunk remained for 198/200 IP questions and 57/58 mixed-domain questions with a grade >= 1 result. Having some text from a relevant chunk does not establish that the answer-bearing sentence survived. The quote counts depend on one Qwen-selected quote per grade-2 pair; other valid support may remain even when that quote is absent.

The five-document limit is the largest loss path (475 IP and 90 mixed-domain grade-2 quotes). Loss inside a selected document is smaller (124 and 34). Pair counts do not rank the levers, because most of those quotes are extra support on questions that already have one visible. Counted by questions, 24 of 195 IP and 8 of 56 mixed-domain questions have a verified grade-2 quote in the top 10 but none visible (under the judge verifier's quote normalization, with credit only inside the quote's own document; `priority1-evidence-budget-report.md`). Label-aware oracles bound what a selector could recover. On IP, better excerpts alone recover 16 of the 24 and better document choice alone recovers 16 (10 by either lever, 22 by at least one, all 24 together). On the mixed set, excerpts recover 5 of the 8, document choice 2, and both together 8. Neither lever dominates on IP, and excerpts lead on the mixed set.

## Model-visible evidence, both runs

The historical run recorded no evidence, so its top 10 was passed through the same formatter (`prepare_evidence`: five documents, 1,200 characters each), and both runs were scored with the same verified grade-2 quotes. For the current run, the rebuilt evidence equals the captured evidence in all 260 cases. When the pre-search returns text, `_with_legal_evidence` in `langchain_config.py` removes the search tool from the turn, and all 260 questions trigger the pre-search. This evidence is therefore all the statute and case-law text the model has for these questions.

| Dataset and metric | 2026-09-19 | Current | Paired change, current minus prior (95% CI) |
| --- | ---: | ---: | ---: |
| IP, verified grade-2 quote in the evidence | 0.860 | 0.860 | 0.000 [-0.050, +0.050] |
| IP, a grade >= 1 document in the evidence | 0.980 | 0.995 | +0.015 [0.000, +0.035] |
| Mixed domains, verified grade-2 quote in the evidence | 0.800 | 0.783 | -0.017 [-0.083, +0.033] |
| Mixed domains, a grade >= 1 document in the evidence | 0.950 | 0.950 | 0.000 [-0.050, +0.050] |

The quote rows ask whether at least one verified grade-2 quote is visible. That was already true for 86% of IP questions, so the measure cannot register a denser evidence set; it also counts one Qwen-selected quote per pair, so other valid support goes uncounted. The document rows are looser: a document counts as shown when it holds any grade >= 1 chunk. The replay report measures density: of the five documents shown, 3.65 held a relevant chunk in the 2026-09-19 run and 4.02 do in the current run on IP (+0.375, 95% CI [+0.240, +0.515]); on the mixed set the change is -0.067 [-0.183, +0.033].

## Timing and limitations

For the current capture, median / p95 per-leg timings were 117–123 / 125–130 ms for dense search and 21–22 / 24–26 ms for sparse search. Fusion was 2.45 / 3.65 ms; reranking 1.86 / 1.88 s. Total capture time was 727.7 s including warm-up and parity work. These are captured stage timings, not an end-to-end Flask latency comparison against the 2026-09-19 configuration.

The calibration gate was explicitly skipped because independent human labels were not available. The local Qwen model digest was `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`. Therefore all relevance findings are provisional development evidence. The locked 200-question confirmation set was not run. These metrics measure retrieved judged chunks and prepared evidence, not generated answer quality and not classic Recall@k. The `k_fused`-only comparison is in the replay report: it re-fused the captured legs at `k_fused=50` with `leg_floor=3`, reused the recorded rerank scores (reranking is pointwise), and judged the 290 pairs not yet judged.

Detailed machine-readable results, hashes, judge provenance, and official graded reports are alongside this report in `priority1-paired-analysis.json` and `grading/{ip-200-fanout,crossdomain-60-fanout}/`.

## Correction after review (2026-09-29)

An earlier version of this report credited the top-10 change to `k_fused` alone, matched quotes by exact substring, and had no model-visible comparison. The exact test counted the quote visible for 310/1,089 IP and 112/244 mixed-domain pairs, lost to the five-document limit for 505 and 91, absent within a selected document for 274 and 41, and visible for at least one grade-2 quote in 145/195 and 45/56 questions. `analyze_priority1.py` now matches with whitespace collapsed; its output in `priority1-paired-analysis.json` reproduces the corrected figures above, and `priority1-paired-analysis.pre-whitespace-fix.json` keeps the superseded output.

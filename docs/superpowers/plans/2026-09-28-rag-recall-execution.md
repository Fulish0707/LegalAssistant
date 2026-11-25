# Recall evaluation execution record

Status: development results and a provisional AI-rater comparison are recorded; independent human judging and locked confirmation remain pending; no serving changes adopted.

The authorized plan is [the recall test plan](2026-09-28-rag-recall-test-plan.md).
Implementation and artifacts were isolated in the codex/rag-recall-evaluation worktree (removed 2026-11-25). The text reports and analysis scripts are now in `data/eval/reports/recall-plan-2026-09-28/`; the harness code (`recall_*.py`, `eval_recall_stages.py` and its tests) is archived in its `tooling-snapshot/` and was not moved into `src/`, so its imports may not match current main. Large run files were not kept in git.
Starting revision: 5d90cbc20f6f1f2d9968e761369cd8029d0fb0ba.

## Scope and decisions

- The main workspace's existing import mission and user edits are preserved.
- The user confirmed that no unused confirmation questions are available.
  All current results are development/regression evidence. Locked confirmation remains pending.
- Runtime baseline uses the local profile and the existing main-workspace data paths.
- No .env contents, API keys, or browser session identifiers are stored in artifacts.
- A concurrent main-workspace commit (160b94f) changed the chat evaluator and developer guide;
  retrieval/runtime files were unchanged. These experiments remain pinned to the starting revision.
- Qdrant client 1.19.0 reports compatibility warning against server 1.12.1. Preserve this
  environment observation; do not infer tested compatibility with other API features.
- Review identified a normalized-score floor bug in the experimental F7 code. It was
  reproduced by a failing test and fixed before the development run.
- Independent review agents hit their service usage limit. Final independent code-quality
  review remains pending; root is continuing local verification and development measurement.

## Evidence so far

- Existing retrieval baseline: 31 tests passed.
- New stage, metric, evidence parity, immutable output and cache/error tests: 18 passed.
- Live smoke: 10/10 ordered top-10 parity matches, zero retrieval-stage errors.
- Source audit: all 96 distinct reference documents in the 100-question target have processed chunks.
- Inventory: 1,010 questions across four pilots, everyday-150 and crossdomain-60;
  zero duplicate NFC/whitespace-normalized questions and zero empty reference-ID lists.
- Development sample: 100 questions, 96 connected source groups;
  criminal 29, civil 22, administrative 29, IP 20 after the everyday sample is included.
- Development experiment records original eight-leg rankings, fusion, reranking and Flask
  evidence preparation. Alternative fusion results are paired on the same candidates.

## Artifact locations

All paths below are relative to that worktree; the text reports among them are now in this repository:

- data/eval/reports/recall-plan-2026-09-28/audit/
- data/eval/reports/recall-plan-2026-09-28/smoke/
- data/eval/reports/recall-plan-2026-09-28/development/

The development directory includes an exact source snapshot matching the code loaded for
that run. Subsequent runner error-recording fixes do not alter the running process.

## Development gates

The baseline, fusion variants, exact/approximate ANN diagnostic, HNSW effort checks, depth 50/200 trials, constrained expansion C5, document-aware selection R3, and shared blind pool are complete. Independent relevance judging, final candidate selection, and locked confirmation remain pending.

## Completed development measurements (2026-09-28)

- Baseline 100-question capture completed with 20/20 exact ordered top-10 parity matches, zero retrieval-stage errors. The equal-domain macro known-reference chunk-cutoff R@10 was 0.455. This is weak source-ID coverage, not semantic Recall@10.
- Full union reranking on 40 fixed questions had paired known-reference R@10 delta +0.000 against baseline (95% exploratory interval [-0.075, +0.075]). None of the F0–F7 variants improved that metric on the 100-question sample. R0 without the cross-encoder was lower by 0.056. No finalist was selected because passage relevance is unjudged.
- Baseline complete known-reference losses: criminal union/fusion/rerank/evidence 2/0/4/1, civil 17/3/1/0, administrative 2/0/0/2, IP 15/4/0/0. All 96 known source documents are in the processed corpus; these counts do not establish loss of all relevant evidence.
- C0 exact-versus-approximate dense search: 40 questions × four dense legs. Mean top-100 overlap was 0.943. C1 hnsw_ef=256 reached 0.983; C2 hnsw_ef=512 reached 0.993. Mean per-dense-search latencies were 7.36 ms, 8.81 ms and 10.53 ms, respectively; exact search was 17.71 ms. None recovered a missing known source ID. Reviewed relevance rescues are unavailable.
- C3/C4 depth experiments on the same 100 questions kept fused budget 100: depth 50 had macro known-reference R@10 0.426 (paired delta -0.029, exploratory 95% CI [-0.065, 0.000]); depth 200 had 0.466 (delta +0.011, CI [0.000, 0.034]). Union source coverage was 0.585/0.585/0.622 for depths 100/50/200. These were separate live searches and requested depth can change ANN ranking.
- Built a blind development judging pool with 3,600 unique question–passage pairs from baseline top 20, all variant top 10, and all known-source chunks. It remains unjudged. Review only review-pool.jsonl; pool-provenance.jsonl contains hidden variant membership.
- Verification: backend suite 563 passed, 3 skipped; Flask evidence suite 17 passed. The added Flask regression verifies grouping and 1,200-character truncation parity. Both suites ran in their respective project environments.

## Further development measurements (2026-09-28)

- C5 generated terms with local qwen3:4b, temperature 0. The original query was joined deterministically to preserve facts and citations. Of 100 first outputs, 52 passed validation, 47 proposed a new article/case identifier and fell back, and one exceeded 30 terms. Two of 20 repeated generations changed the accepted output. All 48 fallback searches reproduced the original-only depth-50 top-10 order.
- C5 held each original/expanded leg at depth 50 and the fused budget at 100. Macro known-source R@10 was 0.435: paired delta -0.020 versus B0 and +0.009 versus the depth-50 control. Union source coverage remained 0.585. Mean expansion generation was 562.5 ms; instrumented evaluator p50/p95 total was 2,973.9/3,345.8 ms including capture serialization. These are not serving latency estimates.
- R3 document-aware selection from the frozen reranked top 100 had macro known-source R@10 0.429, paired delta -0.026 against B0. Supporting-passage survival is unknown without reviewed spans.
- The final blind pool has 3,692 distinct question–passage pairs: 3,600 initial, 38 new C5 top-10, and 54 new R3 top-10. The final calibration sample has 100 pairs, 25 per domain, with sampling provenance separate from the reviewer CSV. Earlier calibration directories are superseded by calibration-100-final/.
- The independent-review gate is fixed before labels: all 100 grades must be valid, every model grade must parse, and Cohen kappa for grade >=1 must be at least 0.60. Per-domain confusion matrices are required. The local qwen3:8b (think on) generated valid judgments for all 100 calibration pairs with zero parse failures and exact pair coverage. Thirteen proposed grade-2 outputs were downgraded to grade 1 because their supporting quote did not verify; no retained grade-2 output lacked a verified quote. Model digest 500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41 and prompt hash 7a2b826ac9cb93494241c98ccc829755d2af0ee37ebd8e3178cdbd798ecb596c are recorded. All grades remain uncalibrated until independent labels establish the predeclared kappa gate.
- Current regression evidence: backend 574 passed, 3 skipped; Flask legal evidence 17 passed. No serving configuration changed.

## Provisional AI-rater comparison (2026-09-28)

- GPT-6 Codex graded a separate copy of all 100 frozen calibration pairs. The blank source CSV was preserved; the AI-labeled copy, reviewer metadata, and calculation are under `data/eval/reports/recall-plan-2026-09-28/ai-calibration-gpt6/`. The grade counts were 29/22/49 for grades 0/1/2.
- The project agreement command matched all 100 pair IDs with no missing Qwen grades. Binary relevant agreement (grade >= 1) was 0.890 and Cohen's κ was 0.719, above the predeclared 0.60 threshold. This is a numerical pass for the GPT-6-to-Qwen AI comparison only. The first three Qwen outputs had been visible during schema inspection before those three GPT-6 labels were assigned, so the full comparison is not fully blind. Excluding those pairs leaves 97 pairs with relevant-label κ 0.701, exact agreement 0.649, weighted κ 0.543, and grade-2-vs-lower κ 0.424. Exact three-grade agreement on all 100 pairs was 0.660, weighted κ was 0.563, and grade-2-vs-lower κ was 0.442.
- Per-domain binary relevant κ was criminal 0.531, civil 1.000, administrative 0.525, and IP 1.000 (25 pairs per domain). These are diagnostics; the civil/IP values come from small samples with high positive prevalence, while criminal/administrative agreement is below 0.60.
- The comparison is not an independent human review. `judge-calibration-100/manifest.json` remains `uncalibrated_complete`; Qwen judgments are not calibrated, and the shared judging pool remains unjudged. This agreement result measures rater consistency only; it does not establish semantic Recall@k or an improvement in retrieval quality. The 200-question locked confirmation also remains unavailable because no unused questions are available.

## Retrieval-quality follow-up (2026-09-29)

- Existing graded development reports show a strong IP-only versus fan-out difference on 200 IP questions: Precision@10 0.689 to 0.824 (paired delta +0.136, 95% CI [+0.103, +0.169]), nDCG@10 0.584 to 0.712 (+0.127, [+0.093, +0.162]), and grade-2 Success@5 0.900 to 0.955 (+0.055, [+0.015, +0.095]). Grade-at-least-1 Success@5 changed by +0.020 ([0.000, +0.045]). These remain provisional Qwen3:8b labels; the IP pool had four failed top-10 parses across the paired runs.
- On 60 mixed-domain questions, oracle corpus routing did not show a clear gain over fan-out: Precision@10 delta +0.013 (95% CI [-0.018, +0.049]), grade-at-least-1 Success@5 +0.017 ([-0.033, +0.069]), and nDCG@10 -0.019 ([-0.056, +0.018]). This does not support automatic routing.
- The existing cross-encoder improves judged IP retrieval over hybrid fusion: Precision@10 +0.066 ([+0.040, +0.092]), nDCG@10 +0.057 ([+0.029, +0.085]), and grade-at-least-1 Success@5 +0.040 ([+0.010, +0.075]).
- A fresh depth-400 development run increased macro known-source candidate coverage from 0.585 at depth 100 to 0.653 (paired delta +0.069, 95% CI [+0.025, +0.120]); known-source R@10 remained 0.466, unchanged from depth 200. Warm captured-stage p50/p95 was 2.59/2.89 seconds versus 2.31/2.59 seconds at depth 200; these are not end-to-end serving latency.
- Offline union of B0 and C5 changed the top-10 order for 50/100 questions but did not improve known-source R@10 (0.455, paired delta 0). These cached cross-run scores are approximate; no positive finding was observed.
- The IP run filtered all four configured corpora with `legal_domain=intellectual_property`; its top-10 and top-50 rankings exactly matched the unfiltered `ip-v1`-only run for all 200 questions. Both improved judged metrics over unfiltered fan-out. The user-facing Flask `/chat` instead sends the original message to `/v1/search` (`k=10`, no filter), then groups at most five documents and clips each quote to 1,200 characters. It does not call the FastAPI `/v1/chat` analyzer. The qwen3:4b domain probe mapped 49/60 mixed-domain questions, with only 29/60 correct overall and 3/20 criminal routes correct; it is evidence against reusing that analyzer for hard routing. Probe artifacts are in `data/eval/reports/recall-plan-2026-09-28/domain-inference-probe-2026-09-29/`. The next evaluation should capture the current local search settings (`k_fused=100`, `leg_floor=3`) and measure visible evidence after Flask grouping and truncation; no serving behavior has changed.
- Artifacts: `data/eval/reports/recall-plan-2026-09-28/recall-quality-update/`, `depth400-mainenv/`, and `candidate-union-expansion/`. Graded source reports are in the main workspace under `data/eval/relevance/ip-200-fanout/` and `crossdomain-60-fanout/`.
- `Success@k` is judged hit rate, not corpus-level Recall@k. The independent human-label and unused 200-question confirmation gates remain pending.

## Current decision and remaining gates

Known-source diagnostics do not select a winner. The principal baseline losses remain at candidate retrieval for civil and IP. Neither C5 nor R3 improved the weak known-source macro metric against B0. R1/R2 and evidence-window survival need independently validated supporting passages; no model or retrieval change is adopted. GPT-6 and Qwen3:8b passed the overall binary agreement threshold as AI raters, but human independent labels are still required before the judge can be calibrated or the shared pool can be scored. The locked 200-question confirmation also remains unavailable; the user confirmed no unused confirmation questions. The mission remains active with T3 pending.

Artifacts are under data/eval/reports/recall-plan-2026-09-28/: c5-report/, r3-report/, judging-pool-final/, calibration-100-final/, judge-calibration-100/ (uncalibrated complete), and ai-calibration-gpt6/ (provisional AI-rater comparison), in addition to the earlier run directories. Generated outputs are local and gitignored.

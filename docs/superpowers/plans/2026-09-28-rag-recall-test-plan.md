# RAG Recall Evaluation Test Plan

**Goal:** Identify and reduce loss of relevant evidence during candidate retrieval, fusion, reranking, and Flask evidence preparation.

**Status:** Planned experiments; no evaluation runs or implementation changes have been performed for this plan.

**Architecture:** Exercise the same unfiltered four-corpus retrieval used by Flask through /v1/search. Record request-local intermediate rankings so candidate selection and reranking can be compared independently. Use local artifacts; LangSmith is not required.

**Tech stack:** Python 3.11, Qdrant, Ollama bge-m3, BM25, weighted RRF, BAAI/bge-reranker-v2-m3, existing legal_rag evaluation tools.

**Date:** 2026-09-28. Thresholds below are proposed experiment acceptance rules, not previously agreed product requirements.

## 1. Baseline and scope

Freeze B0 from the resolved local profile, including environment overrides:

| Component | Expected baseline |
| --- | --- |
| Corpora | criminal-v1, civil-v1, administrative-v1, ip-v1 |
| Domain/type filters | None for the primary web-flow evaluation |
| Embedding | Ollama bge-m3; query context 1024 |
| Sparse tokenizer | regex with legal identifiers and Hangul bigrams |
| Leg depths | 100 dense and 100 sparse per corpus |
| Fusion | RRF constant 60; dense weight 1; sparse weight 0.75; leg floor 3 |
| Fused candidates | 100 |
| Reranker | BAAI/bge-reranker-v2-m3, CUDA, float16, max_length 512, batch size 16 |
| Search response | 10 chunks for Flask |
| Flask evidence | At most 5 documents; joined excerpts truncated to 1,200 characters per document |

Verify resolved values rather than trusting YAML alone. Record package versions, Git revision and dirty diff, model digests/cache revisions, index manifests, tokenizer, hardware, corpus counts, and configuration hashes. Never copy .env or secrets into reports.

Primary scope: original query → eight retrieval legs → fusion → rerank → top 10 chunks → Flask evidence.
The /v1/chat rewriting/orchestration path is a separate regression track; results from it do not establish web-flow quality.

Existing evidence to respect:
- Structure-aware ip-v2 rechunking did not justify adoption.
- A prior civil query lost a strong candidate at fusion rank 129.
- Equal-weight fusion performed worse than dense in the single-domain IP experiment.
- Increasing reranker input length previously had uncertain benefit and increased cost.
- Existing pilot sets have been used for tuning. They are development/regression data.

Source: [retrieval diagnostics](../../retrieval-diagnostics.md).

## 2. Datasets and labels

### 2.1 Development and regression

- [ ] Inventory and hash the four data/eval/*-pilot-200.jsonl files, data/eval/targets/crossdomain-60/target.jsonl, and data/eval/everyday-150.jsonl.
- [ ] Check schema, corpus availability, duplicate questions and source-document overlap before sampling.
- [ ] Create a 100-question development target: 20 from each domain's pilot and 20 from everyday-150. Use seed 20260928; preserve source IDs and record sampled membership.
- [ ] Deduplicate normalized questions before sampling. Normalize Unicode NFC and whitespace; preserve punctuation, dates and article numbers.
- [ ] Use the remaining historical questions for regression reporting, never call them untouched holdout data.
- [ ] Track reference documents missing from the indexed corpus separately. Report their count against all questions; do not silently remove them from an end-to-end coverage score.

### 2.2 Locked confirmation set

- [ ] Obtain 200 fresh questions: 50 each for criminal, civil, administrative and intellectual-property topics.
- [ ] Within each domain, target 15 explicit-reference lookups, 25 everyday single-issue questions and 10 multi-issue questions.
- [ ] Group paraphrases and questions sharing a source document into the same partition. Check overlap with historical evaluation and any model-training data; document uncertain provenance.
- [ ] Have an independent reviewer check relevance labels and answer-bearing passages. If only AI reference labels are available, label the result provisional and identify the models.
- [ ] Lock membership and hashes before running variants. Choose the final variant using development results only.
- [ ] Do not create a supposed fresh test by merely re-splitting old pilots. If new questions cannot be obtained, issue a development-only result.

The 200-question set is an initial confirmation budget, not a guarantee of statistical power. If uncertainty prevents acceptance, report inconclusive. Any larger confirmation study must be specified before inspecting its results.

### 2.3 Relevance rubric and judging

For each question–passage pair:
- 0: wrong issue or lexical overlap without useful evidence.
- 1: relevant issue, but insufficient to answer the question alone.
- 2: contains a rule, reasoning or factual application sufficient to support an answer to the issue.

Keep source-document identity separate from relevance. Another case can be useful even when it is not the dataset's original source.

Build a shared judging pool from every development variant's top 10, B0 top 20, and known reference passages. Add every newly surfaced top-10 passage before comparing variants. For confirmation, pool only B0 and the locked winner plus known references; blind run identity and rank during judging.

Use the same judge model, prompt hash, truncation rules and reasoning mode throughout a comparison. Calibrate on 100 development pairs, 25 per domain, stratified across easy positives, difficult positives and negatives. Predeclare binary grade >= 1 Cohen kappa >= 0.60 as the automated-judge gate; report per-domain confusion matrices. If it fails, use independent adjudication or revise and recalibrate on a fresh sample. Do not switch agreement statistics after seeing failure.

Require valid judgments for 100% of compared top-10 passages, with zero unresolved parse failures there. Existing coverage checks can count failed parses as covered; this plan applies the stricter gate. Grade-2 claims require independently validated labels at that boundary.

## 3. Metrics: define k and the denominator

| Metric | Definition and use |
| --- | --- |
| Known-reference coverage at a stage | Fraction of labeled relevant document IDs appearing anywhere in that stage's candidates; only as complete as the labels |
| Known-reference chunk-cutoff Recall@10 | Labeled relevant documents present in the first 10 chunks divided by labeled relevant documents |
| Unique-document Recall@10 | Deduplicate document IDs before taking 10; report separately with the retrieved depth |
| Success@10 | At least one judged grade >= 1 passage in the first 10 chunks |
| Sufficient-evidence Success@5 | At least one validated grade-2 passage among the five selected evidence documents |
| Precision@10 | Judged relevant chunks / 10; missing slots count as nonrelevant |
| ANN overlap@100 | Intersection of approximate and exact dense top-100 IDs / 100, or exact-result count if fewer exist |
| Evidence survival | Fraction of questions with sufficient retrieved evidence whose supporting passage remains after Flask preparation |
| Operational metrics | Warm p50/p95 search latency, error rate, peak VRAM/RAM, candidate counts and judgment cost |

Single-positive known-reference recall is Success@k for that source, not coverage of every legally relevant authority. In particular, do not use low IP exact-source recall alone as proof of poor semantic retrieval.

Do not report exhaustive semantic Recall@k from a shallow judging pool. If pooled recall is reported for analysis, freeze its common denominator across variants and label it pool-relative. Never compare nDCG values whose judgment pools differ.

Compute per-domain means and an equal-domain macro average for confirmation. Also report everyday-query and explicit-reference slices. Resample query/source groups, not individual paraphrases, for paired confidence intervals.

## 4. Required measurement tooling

These are proposed additions, not existing commands. Implement them only when executing this plan.

| Proposed file | Responsibility |
| --- | --- |
| src/legal_rag/evaluation/recall_stages.py | Capture per-query leg, fused, reranked and prepared-evidence results without shared mutable last_trace state |
| src/legal_rag/evaluation/recall_variants.py | Explicit experiment definitions, raw-score normalization, stage replay and configuration validation |
| src/legal_rag/evaluation/recall_report.py | Metric definitions above, paired group bootstrap, completeness checks and experiment comparisons |
| scripts/eval_recall_stages.py | CLI for immutable manifests, live capture and offline replay |
| tests/legal_rag/unit/test_recall_stages.py | Stage accounting and ranking invariants |
| tests/legal_rag/unit/test_recall_report.py | Metric denominators, grouping and incomplete-judgment rejection |
| tests/test_legal_search_tool.py | Regression coverage for actual Flask grouping/truncation semantics |

Artifact root: data/eval/reports/recall-plan-2026-09-28/. Every experiment gets a new directory; refuse overwrites.

Required artifacts:
- manifest.json: experiment ID, source hashes, full nonsecret resolved settings, code/model versions, timestamps and hardware.
- target.jsonl: case ID, grouping ID, domain, query, source provenance and reference IDs.
- stages.jsonl: raw per-leg ranks/scores, corpus, chunk/document IDs, fusion scores, rerank scores, requested/actual sizes, elapsed time and errors.
- evidence.jsonl: selected document/chunk IDs and exact source offsets/text retained by Flask; record truncation.
- judgments.jsonl: rubric, reviewer/judge identity, model/prompt hash and grade.
- per_case.csv, summary.json, report.md: paired outcomes, uncertainty, failure categories and decision.

Deep capture may contain source text; keep artifacts local. No credentials, vector arrays or browser session secrets are needed.

Correctness fixtures before expensive experiments:
- [ ] One gold document with three returned chunks counts once in document recall.
- [ ] A grade-2 chunk at fused rank 129 is absent at fused depth 100 and present in full-union replay.
- [ ] Empty successful search and HTTP/search failure produce different statuses.
- [ ] Concurrent query recordings cannot share or overwrite rankings.
- [ ] A supporting sentence after character 1,200 is counted as lost by current Flask preparation.
- [ ] Tied reranker scores preserve input order.
- [ ] Incomplete judgments and failed top-10 judgments block final comparison.
- [ ] Replayed B0 reproduces live B0 top-10 IDs and order on all 20 parity queries.
- [ ] Dense/sparse depths can vary independently: current factory passes max(k_dense, k_sparse) to every leg, so unequal-depth trials require an explicit harness change.

## 5. Execution phases

### Phase A — Establish the baseline and locate loss

- [ ] Run B0 on the 100 development questions, saving every leg at depth 100.
- [ ] Record union membership before fusion, fused top 100, reranked top 10 and prepared evidence.
- [ ] Assign reviewed failures to: absent source, retrieval miss, fusion loss, rerank loss, evidence truncation, or label ambiguity. Allow multiple causes and cite the supporting IDs.
- [ ] Capture a separate depth-200 leg run for replay; do not substitute it for B0. ANN top-100 from a depth-200 request may differ from a depth-100 request.
- [ ] Score the entire depth-100 union with the current cross-encoder on 40 fixed development questions, 10 per domain. Keep the same original query and model inputs.
- [ ] Compare full-union reranking against B0. This estimates avoidable fusion loss under the current reranker, not an oracle relevance ceiling.

Proceed to fusion tuning when reviewed relevant candidates are present in the union but excluded by fusion. Prioritize candidate retrieval when those candidates are absent. Review labels/content first when the apparent misses are ambiguous.

### Phase B — Fusion experiments

Keep query, leg results, model and final k fixed. Replay original eight leg lists, not previously fused dense/sparse output files.

| ID | Changes from B0 |
| --- | --- |
| F0 | leg_floor 0 |
| F1 | leg_floor 5 |
| F2 | k_fused 200, floor 3 |
| F3 | sparse_weight 0.50 |
| F4 | sparse_weight 1.00 |
| F5 | rrf_k 20 |
| F6 | rrf_k 100 |
| F7a/F7b/F7c | Normalized score combination, dense coefficient 0.50/0.75/1.00 |

For F7: normalize each query/corpus/leg's returned scores to [0,1] using min/max; equal-score lists receive 1 for all returned entries. Missing hits contribute 0. A chunk's score is alpha*dense + (1-alpha)*sparse within its corpus, then compare these scores across corpora. Keep floor 3 and candidate budget 100 for this experiment. Record cross-corpus calibration as a limitation; this is a defined hypothesis, not a guaranteed comparable score scale.

- [ ] Cache pointwise cross-encoder scores for identical query/chunk/model/input combinations.
- [ ] Recompute judgments for newly promoted top-10 results before ranking variants.
- [ ] Report stage reference coverage, Success@10, Precision@10 and corpus shares.
- [ ] Choose at most one fusion finalist by development Success@10, then Precision@10, then lower live p95 latency.
- [ ] Measure finalist latency live; offline replay timings do not establish serving latency.

### Phase C — Candidate retrieval experiments

Run each variant against B0 first, not the fusion winner, to isolate its effect.

| ID | Experiment | Interpretation |
| --- | --- | --- |
| C0 | Exact dense search vs current approximate search; 40 fixed questions, 10/domain | Quantify ANN loss |
| C1/C2 | hnsw_ef 256 / 512, depth 100 | Run only if C0 shows missing neighbors |
| C3/C4 | Both leg depths 50 / 200; fused budget remains 100 | Measure depth independently of reranking cost |
| C5 | Original query plus one constrained legal-term expansion | Measure vocabulary mismatch |
| C6 | Separate shadow BM25 index using Kiwi, same documents/chunks | Run only after reviewed lexical failures justify rebuilding |
| C7 | Grounded title/article/section prefixes in shadow dense and sparse indexes | Run only after reviewed missing-context failures justify rebuilding |

C0 exact search can be expensive; record wall time and stop after the fixed sample. If macro ANN overlap@100 >= 0.99 and exact retrieval rescues no reviewed relevant misses, skip C1/C2 for this cycle. This is a prioritization rule, not proof of universally perfect ANN behavior.

For C5, use qwen3:4b with temperature 0 and cache the first generated expansion. Request JSON with one field, legal_query, containing at most 30 whitespace-separated terms. Preserve stated facts and explicit references; forbid invented article/case numbers and hypothetical answers. Reject malformed output, empty output or novel citation identifiers and fall back to the original query. Store the prompt and output hashes. Repeat generation for 20 development queries to report instability.

Original and expanded queries each receive depth 50 per leg; keep the original query for reranking, k_fused 100 and final k 10. Compare both B0 and an original-only depth-50 control. This fixes the nominal total per-leg candidate budget at 100 for the expansion variant, though duplicates reduce its union. Report latency including expansion generation. This path needs new harness support; /v1/search currently uses the original query only.

For C6/C7, preserve original indexes and use new collection names. Rebuild query/index tokenization consistently and retain source-span mappings. Reject experiments that silently mix different tokenizers or embedding representations.

Select at most one candidate-retrieval finalist using the same development ranking rule as Phase B.

### Phase D — Reranking and evidence selection

Use identical saved B0 fused candidates to isolate reranking.

| ID | Experiment |
| --- | --- |
| R0 | No cross-encoder; retain fused order |
| R1 | Current cross-encoder, max_length 1024, same precision |
| R2 | Sliding passage windows scored with current model |
| R3 | Document-aware selection from the same reranked top 100 |

R2: reserve the tokenized query and special-token budget within 512; tokenize passage windows with overlap min(64, floor(window_size/4)); score all windows. Aggregate by maximum window score, breaking ties by original fused rank. Record window count and document length to detect long-document advantage. If the query alone exhausts the budget, record an invalid case rather than silently truncating it. Preserve source offsets for every window.

R3: take the highest-ranked chunk from each of the first five distinct documents; fill the remaining five chunk slots in original reranked order from those documents. Compare document coverage and supporting-passage coverage with B0. This is a selection hypothesis and may lose useful authorities beyond those five documents.

- [ ] Report which previously relevant results improve or regress, and whether the relevant passage was outside the 512-token input.
- [ ] Do not adopt R1 simply because it changes rankings; longer input previously showed uncertain gains.
- [ ] Select at most one reranking/selection finalist.
- [ ] Compare current 1,200-character prefix evidence against a 1,200-character sentence-bounded excerpt centered on the best scored source window. Apply the same document count and total evidence budget.
- [ ] Measure evidence survival using validated supporting spans. Do not call an evidence-formatting gain a retrieval Recall@10 gain.

Defer reranker fine-tuning and replacement models to another cycle unless the above work produces enough reliable positive/negative labels and clearly identifies model-ranking errors.

### Phase E — Combine and confirm

- [ ] Assemble one combined finalist from the independently selected changes.
- [ ] Run B0, each selected component alone, and the combination on development data to detect interactions.
- [ ] Freeze the combined configuration, code, judge and acceptance rules before opening confirmation results.
- [ ] Evaluate only B0 and the frozen combination on the 200-question locked set.
- [ ] Run the historical regression set separately and report it as previously exposed data.
- [ ] Keep the serving configuration unchanged until acceptance criteria pass.

## 6. Acceptance and stopping rules

Proposed final acceptance requires all of:

1. Macro Success@10 at grade >= 1 improves by at least 0.03 absolute, and the paired 95% bootstrap lower bound is above zero.
2. No domain's observed Success@10 drops more than 0.05; additionally, its paired 95% lower bound must exceed -0.10. If the 50-question domain sample cannot establish that, mark domain safety inconclusive.
3. Macro Precision@10 paired lower bound exceeds -0.02.
4. On the explicit-reference subset, known-reference chunk-cutoff Recall@10 paired lower bound exceeds -0.02. Review every newly missed named reference.
5. Evidence survival does not regress; every newly lost validated supporting passage is reviewed.
6. Warm p95 search latency is at most 1.25 times B0, and there are no new OOMs or search errors.
7. Judgment completeness, parity and provenance checks pass.

Use 10,000 paired bootstrap resamples with seed 20260928, stratified by domain and clustered by grouping ID. Report point estimates and intervals, including negative changes. If any interval is inconclusive, retain B0 and report the limitation; do not reinterpret a failed gate after seeing results.

Development comparisons are exploratory. Only one frozen combination enters confirmation, avoiding selection of a winner from repeated holdout tests.

Latency procedure: 10 unmeasured warm-up queries, then the same fixed 100 development queries for B0 and finalist in alternating configuration blocks; repeat three times, reverse block order on the second repetition. Run one heavy workload at a time. Measure client round-trip time and stage time separately; include query expansion where applicable. Report cold-start latency separately. Keep model residency and cache conditions consistent.

Before a large run, time 10 questions and estimate retrieval and new-judgment cost separately. Do not extrapolate old fp32 timings to current fp16 execution. If resource limits prevent a prescribed comparison, mark it unmeasured rather than reducing the test set after inspecting quality.

## 7. Existing commands and limitations

The following command is already supported. Run from the project root after creating the report directory; this is an initial known-reference smoke evaluation, not the stage-aware confirmation study:

~~~bash
mkdir -p data/eval/reports/recall-plan-2026-09-28
uv run python scripts/eval_retrieval.py \
  --profile local \
  --dataset data/eval/criminal-pilot-200.jsonl \
  --json data/eval/reports/recall-plan-2026-09-28/criminal-baseline.json
~~~

Existing reusable tools:
- scripts/diagnose_retrieval.py: records diagnostic cases; --limit limits exported review cases, not total queries, and its domain filter differs from unfiltered serving.
- scripts/build_fanout_pool.py: supports unfiltered corpus fan-out; explicitly set --run-depth 100 with the current local profile. Requires QA sources matching target IDs.
- scripts/sweep_fusion_weights.py: existing replay shortcut is valid for its single-corpus setup, not eight raw legs.
- scripts/rerank_relevance_pool.py: useful for single-corpus pools; currently loads chunks from one processed corpus and does not preserve all serving reranker settings automatically.
- scripts/eval_chat_pipeline.py: evaluates /v1/chat components, not Flask evidence preparation.
- scripts/judge_agreement.py and scripts/eval_graded.py: reusable judging/report tools, subject to the stricter completeness and label rules above.

Do not invent CLI flags for F0–R3. The stage runner in Section 4 must exist and pass parity checks before those experiments can execute.

## 8. Final deliverable checklist

- [ ] Baseline and target manifests with immutable hashes.
- [ ] Stage loss table for each domain and query type.
- [ ] Per-variant candidate coverage, Success@10, Precision@10, reference recall and evidence survival.
- [ ] Paired confidence intervals and latency/VRAM comparison.
- [ ] Reviewed improvements and regressions with source IDs and supporting spans.
- [ ] Explicit list of incomplete labels, excluded sources, unrun experiments and resource limits.
- [ ] Decision: adopt, reject, or inconclusive; link the exact winning configuration only if accepted.
- [ ] No claim that LangSmith, higher k, or a different model improved quality without measured evidence.

## 9. Research informing the hypotheses

- [Qdrant candidate-depth guidance](https://qdrant.tech/documentation/search-tuning/candidate-depth/): distinguish result depth from ANN traversal effort.
- [Analysis of Fusion Functions](https://arxiv.org/abs/2210.11934): motivates testing score combinations against RRF.
- [Query2doc](https://aclanthology.org/2023.emnlp-main.585/): motivates constrained query-expansion experiments; its gains are not predictions for this corpus.
- [Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval): motivates testing contextual indexing on a limited shadow corpus.
- [BGE reranker model card](https://huggingface.co/BAAI/bge-reranker-v2-m3): current model reference.
- [Sentence Transformers reranker training](https://www.sbert.net/examples/cross_encoder/training/rerankers/README.html): later-stage domain training reference.

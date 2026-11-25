# Retrieval quality follow-up — 2026-09-29

This update uses existing graded relevance reports from the main workspace plus new depth-400 and candidate-union development runs from the isolated worktree. No retrieval configuration or serving code changed.

## Graded semantic results

The grader was `ollama:qwen3:8b` with prompt hash `7a2b826a9cb9`. These are provisional model judgments: the 100-pair agreement check was AI-to-AI, not an independent human review. The IP fan-out pool has three failed top-10 parses and its single-corpus run has one; failures were counted as nonrelevant. The mixed-domain pool has no top-10 parse failures.

| Comparison | Precision@10 | Success@5 (grade ≥1) | Success@5 (grade 2) | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| IP fan-out | 0.689 | 0.970 | 0.900 | 0.584 |
| IP-only | 0.824 | 0.990 | 0.955 | 0.712 |

For the 200 IP questions, IP-only minus fan-out was **+0.136 Precision@10** (paired 95% CI [+0.103, +0.169]), **+0.127 nDCG@10** ([+0.093, +0.162]), and **+0.055 grade-2 Success@5** ([+0.015, +0.095]). Grade ≥1 Success@5 increased by 0.020, with CI [0.000, 0.045]. Excluding the four cases with failed judgments still gave +0.135 Precision@10 and +0.128 nDCG@10.

Routing every query by a known domain did not show the same clear result on the 60 mixed-domain cases. Oracle routing changed Precision@10 by +0.013 (95% CI [−0.018, +0.049]), grade ≥1 Success@5 by +0.017 ([−0.033, +0.069]), and nDCG@10 by −0.019 ([−0.056, +0.018]) against fan-out. These intervals include zero.

The cross-encoder already improves the IP top-10 retrieval results over hybrid fusion: grade ≥1 Success@5 rose from 0.945 to 0.985 (paired delta +0.040; 95% CI [+0.010, +0.075]), Success@10 rose by +0.020 ([+0.005, +0.040]), Precision@10 by +0.066 ([+0.040, +0.092]), and nDCG@10 by +0.057 ([+0.029, +0.085]).

## Automatic domain inference probe

The existing `QueryAnalyzer` was run with `ollama:qwen3:4b` at temperature 0 on the 60 cross-domain development questions (20 each for criminal, civil, and administrative). The analysis prompt returns Korean domain labels; `민사`, `형사`, and `행정` were mapped to `civil`, `criminal`, and `administrative`. No chat history was supplied. All 60 calls returned structured analysis, but the result measures label prediction only, not retrieval quality.

| Target domain | Cases | Mapped | Correct route | Wrong route | Unmapped |
| --- | ---: | ---: | ---: | ---: | ---: |
| Criminal | 20 | 19 | 3 | 16 | 1 |
| Civil | 20 | 12 | 12 | 0 | 8 |
| Administrative | 20 | 18 | 14 | 4 | 2 |
| **Total** | **60** | **49** | **29** | **20** | **11** |

Overall route accuracy was 29/60 (48.3%); among mapped cases it was 29/49 (59.2%). The prompt permits family, labor, and commercial labels that have no matching configured corpus, and it does not include intellectual property. There is no confidence field to support a reliable abstention threshold. A hard domain filter based on the current analysis would risk excluding the correct corpus. The probe is recorded at `data/eval/reports/recall-plan-2026-09-28/domain-inference-probe-2026-09-29/`.

## Candidate-depth and expansion replays

| Dense/sparse depth | Macro known-source candidate coverage | Macro known-source R@10 |
| ---: | ---: | ---: |
| 100 | 0.585 | 0.455 |
| 200 | 0.622 | 0.466 |
| 400 | 0.653 | 0.466 |

Depth 400 improved candidate-union source coverage over depth 100 by 0.069 (95% CI [+0.025, +0.120]) but did not improve R@10 over depth 200 (paired delta 0.000). Warm captured-stage p50/p95 was 2.59/2.89 seconds at depth 400 versus 2.31/2.59 seconds at depth 200; these are not end-to-end serving latency measurements.

The baseline-plus-C5 candidate union changed the top-10 order in 50 of 100 questions but left macro known-source R@10 at 0.455, equal to B0. C5 alone scored 0.435. This did not demonstrate a recall improvement.

## Priority 1 — current Flask path measured

The current local path was captured and compared with the historical `k_fused=50` fan-out run on the same 200 IP and 60 mixed-domain development questions. The resolved path uses `k_dense=k_sparse=k_fused=100` and `leg_floor=3`; all top-10 pairs were judged under the same Qwen prompt with zero parse failures.

On IP, grade >= 1 Precision@10 increased by 0.101 (95% CI [+0.074, +0.128]) and grade-2 Success@10 by 0.030 ([+0.010, +0.055]). On mixed domains, grade >= 1 and grade-2 Success@10 were unchanged; grade >= 1 Precision@10 increased by 0.012 with an interval crossing zero. Grade-2 Success@5 fell by 0.050, concentrated in civil and criminal questions. This supports the current k=100 ranking on IP but does not show a mixed-domain Success@10 gain.

The Flask five-document / 1,200-character output retained the selected Qwen grade-2 quote for 310/1,089 IP pairs and 112/244 mixed-domain pairs. Of the remaining verified quotes, 505 IP and 91 mixed-domain source documents were beyond the first-five-document limit; 274 and 41 were absent within selected-document excerpts. This points to document selection as the next component to isolate while holding retrieval fixed.

These are provisional model-judged results from reused development data. The judge's independent human-calibration gate was skipped, no fresh confirmation questions were available, and the historical targets are not exhaustive relevant-source sets. The comparison therefore does not establish classic corpus-level Recall@k or answer quality. Full paired tables, intervals, provenance, and caveats are in [the Priority 1 report](../live-flask-current-2026-09-29-rootvenv/priority1-report.md).
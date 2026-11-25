# Preserving baseline candidates while adding C5 candidates

Offline paired replay on the same 100 previously used development questions. The candidate set is the union of the production B0 top 100 and the C5 original-plus-expansion top 100. Shared candidates keep B0 cross-encoder scores; C5-only candidates use the C5 scores. This is a cached pointwise replay, not a serving run.

| Run | Macro known-source R@10 |
| --- | ---: |
| B0 | 0.455 |
| C5 | 0.435 |
| B0 + C5 candidate union | 0.455 |

The union changed the exact top 10 in 50 of 100 questions and did not improve macro known-source R@10 over B0 (paired delta +0.000, 95% CI [+0.000, +0.000]). It adds a mean 24.7 C5-only candidates per case to a mean union size of 124.7.

The shared cross-encoder score median absolute drift is 0.000000, p95 0.000000, maximum 0.001907. The baseline-score replay preserved B0's top 10 in all 100 cases, so this experiment found no exact-reference gain.

## Limits

- Exact source IDs are incomplete relevance labels; this does not establish semantic Recall@10.
- Scores were collected in separate runs. A positive result would need a fresh single-process rerank over the union.
- Reranking a larger set changes cost; latency was not measured.

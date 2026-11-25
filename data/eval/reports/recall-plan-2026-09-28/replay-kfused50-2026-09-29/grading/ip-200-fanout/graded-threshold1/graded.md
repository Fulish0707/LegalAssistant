# Graded retrieval comparison

- Judge: `ollama:qwen3:8b`, prompt `7a2b826ac9cb`, failed parses: 0, out-of-pool judgments ignored: 0
- Pool: pool_depth 10, settings_hash `replay-k50-floor3-recorded-fp16-scores`
- Headline relevance: grade ≥ 1 (success@k, mrr, precision@10, ndcg@10_binary)
- Corpora per run: `previous_k50` criminal-v1 + civil-v1 + administrative-v1 + ip-v1, `flask_current_k100` criminal-v1 + civil-v1 + administrative-v1 + ip-v1; filters None.

## chunk

| Metric | flask_current_k100 | previous_k50 | replay_k50 |
|---|---:|---:|---:|
| cases | 200 | 200 | 200 |
| cases_without_relevant_binary | 0 | 0 | 0 |
| ndcg@10_binary | 0.842 | 0.744 | 0.773 |
| success@5 | 0.990 | 0.970 | 0.965 |
| success@10 | 1.000 | 0.990 | 0.990 |
| mrr@10 | 0.885 | 0.851 | 0.863 |
| precision@10 | 0.790 | 0.690 | 0.718 |
| mean_judged_coverage@10 | 1.000 | 1.000 | 1.000 |
| min_judged_coverage@10 | 1.000 | 1.000 | 1.000 |
| failed_top10_total | 0 | 0 | 0 |

### Reference (graded)

| Metric | flask_current_k100 | previous_k50 | replay_k50 |
|---|---:|---:|---:|
| cases_without_relevant | 0 | 0 | 0 |
| ndcg@10 | 0.732 | 0.629 | 0.669 |
| precision@10_grade1 | 0.790 | 0.690 | 0.718 |
| precision@10_grade2 | 0.544 | 0.449 | 0.490 |

## document

| Metric | flask_current_k100 | previous_k50 | replay_k50 |
|---|---:|---:|---:|
| cases | 200 | 200 | 200 |
| cases_without_relevant_binary | 0 | 0 | 0 |
| ndcg@10_binary | 0.849 | 0.740 | 0.776 |
| success@5 | 0.995 | 0.985 | 0.985 |
| success@10 | 1.000 | 0.995 | 0.990 |
| mrr@10 | 0.903 | 0.874 | 0.882 |
| precision@10 | 0.778 | 0.660 | 0.701 |
| mean_judged_coverage@10 | 1.000 | 1.000 | 1.000 |
| min_judged_coverage@10 | 1.000 | 1.000 | 1.000 |
| failed_top10_total | 0 | 0 | 0 |

### Reference (graded)

| Metric | flask_current_k100 | previous_k50 | replay_k50 |
|---|---:|---:|---:|
| cases_without_relevant | 0 | 0 | 0 |
| ndcg@10 | 0.744 | 0.637 | 0.680 |
| precision@10_grade1 | 0.778 | 0.660 | 0.701 |
| precision@10_grade2 | 0.540 | 0.439 | 0.487 |

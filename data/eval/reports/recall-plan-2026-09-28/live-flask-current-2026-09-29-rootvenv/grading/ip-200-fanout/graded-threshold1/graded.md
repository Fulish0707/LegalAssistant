# Graded retrieval comparison

- Judge: `ollama:qwen3:8b`, prompt `7a2b826ac9cb`, failed parses: 0, out-of-pool judgments ignored: 0
- Pool: pool_depth 10, settings_hash `live-k100-floor3-crossencoder-fp16-512`
- Headline relevance: grade ≥ 1 (success@k, mrr, precision@10, ndcg@10_binary)
- Corpora per run: `previous_k50` criminal-v1 + civil-v1 + administrative-v1 + ip-v1, `flask_current_k100` criminal-v1 + civil-v1 + administrative-v1 + ip-v1; filters None.

## chunk

| Metric | flask_current_k100 | previous_k50 |
|---|---:|---:|
| cases | 200 | 200 |
| cases_without_relevant_binary | 0 | 0 |
| ndcg@10_binary | 0.849 | 0.749 |
| success@5 | 0.990 | 0.970 |
| success@10 | 1.000 | 0.990 |
| mrr@10 | 0.885 | 0.851 |
| precision@10 | 0.790 | 0.690 |
| mean_judged_coverage@10 | 1.000 | 1.000 |
| min_judged_coverage@10 | 1.000 | 1.000 |
| failed_top10_total | 0 | 0 |

### Reference (graded)

| Metric | flask_current_k100 | previous_k50 |
|---|---:|---:|
| cases_without_relevant | 0 | 0 |
| ndcg@10 | 0.748 | 0.641 |
| precision@10_grade1 | 0.790 | 0.690 |
| precision@10_grade2 | 0.544 | 0.449 |

## document

| Metric | flask_current_k100 | previous_k50 |
|---|---:|---:|
| cases | 200 | 200 |
| cases_without_relevant_binary | 0 | 0 |
| ndcg@10_binary | 0.856 | 0.744 |
| success@5 | 0.995 | 0.980 |
| success@10 | 1.000 | 0.990 |
| mrr@10 | 0.903 | 0.872 |
| precision@10 | 0.777 | 0.659 |
| mean_judged_coverage@10 | 1.000 | 1.000 |
| min_judged_coverage@10 | 1.000 | 1.000 |
| failed_top10_total | 0 | 0 |

### Reference (graded)

| Metric | flask_current_k100 | previous_k50 |
|---|---:|---:|
| cases_without_relevant | 0 | 0 |
| ndcg@10 | 0.758 | 0.646 |
| precision@10_grade1 | 0.777 | 0.659 |
| precision@10_grade2 | 0.539 | 0.438 |

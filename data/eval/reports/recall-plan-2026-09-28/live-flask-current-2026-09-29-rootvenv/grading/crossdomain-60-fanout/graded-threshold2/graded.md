# Graded retrieval comparison

- Judge: `ollama:qwen3:8b`, prompt `7a2b826ac9cb`, failed parses: 0, out-of-pool judgments ignored: 0
- Pool: pool_depth 10, settings_hash `live-k100-floor3-crossencoder-fp16-512`
- Headline relevance: grade ≥ 2 (success@k, mrr, precision@10, ndcg@10_binary)
- Corpora per run: `previous_k50` criminal-v1 + civil-v1 + administrative-v1 + ip-v1, `flask_current_k100` criminal-v1 + civil-v1 + administrative-v1 + ip-v1; filters None.

## chunk

| Metric | flask_current_k100 | previous_k50 |
|---|---:|---:|
| cases | 60 | 60 |
| cases_without_relevant | 2 | 2 |
| cases_without_relevant_binary | 4 | 4 |
| ndcg@10 | 0.802 | 0.804 |
| ndcg@10_binary | 0.772 | 0.775 |
| success@5 | 0.867 | 0.917 |
| success@10 | 0.933 | 0.933 |
| mrr@10 | 0.744 | 0.758 |
| precision@10 | 0.407 | 0.402 |
| precision@10_grade1 | 0.592 | 0.580 |
| precision@10_grade2 | 0.407 | 0.402 |
| mean_judged_coverage@10 | 1.000 | 1.000 |
| min_judged_coverage@10 | 1.000 | 1.000 |
| failed_top10_total | 0 | 0 |

## document

| Metric | flask_current_k100 | previous_k50 |
|---|---:|---:|
| cases | 60 | 60 |
| cases_without_relevant | 2 | 2 |
| cases_without_relevant_binary | 4 | 4 |
| ndcg@10 | 0.851 | 0.853 |
| ndcg@10_binary | 0.830 | 0.834 |
| success@5 | 0.883 | 0.917 |
| success@10 | 0.933 | 0.933 |
| mrr@10 | 0.810 | 0.823 |
| precision@10 | 0.393 | 0.390 |
| precision@10_grade1 | 0.550 | 0.537 |
| precision@10_grade2 | 0.393 | 0.390 |
| mean_judged_coverage@10 | 1.000 | 1.000 |
| min_judged_coverage@10 | 1.000 | 1.000 |
| failed_top10_total | 0 | 0 |

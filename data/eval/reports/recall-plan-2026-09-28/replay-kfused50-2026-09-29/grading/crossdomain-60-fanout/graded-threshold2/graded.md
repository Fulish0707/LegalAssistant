# Graded retrieval comparison

- Judge: `ollama:qwen3:8b`, prompt `7a2b826ac9cb`, failed parses: 0, out-of-pool judgments ignored: 0
- Pool: pool_depth 10, settings_hash `replay-k50-floor3-recorded-fp16-scores`
- Headline relevance: grade ≥ 2 (success@k, mrr, precision@10, ndcg@10_binary)
- Corpora per run: `previous_k50` criminal-v1 + civil-v1 + administrative-v1 + ip-v1, `flask_current_k100` criminal-v1 + civil-v1 + administrative-v1 + ip-v1; filters None.

## chunk

| Metric | flask_current_k100 | previous_k50 | replay_k50 |
|---|---:|---:|---:|
| cases | 60 | 60 | 60 |
| cases_without_relevant | 2 | 2 | 2 |
| cases_without_relevant_binary | 4 | 4 | 4 |
| ndcg@10 | 0.795 | 0.798 | 0.804 |
| ndcg@10_binary | 0.766 | 0.768 | 0.784 |
| success@5 | 0.867 | 0.917 | 0.900 |
| success@10 | 0.933 | 0.933 | 0.933 |
| mrr@10 | 0.744 | 0.758 | 0.764 |
| precision@10 | 0.407 | 0.402 | 0.403 |
| precision@10_grade1 | 0.592 | 0.580 | 0.580 |
| precision@10_grade2 | 0.407 | 0.402 | 0.403 |
| mean_judged_coverage@10 | 1.000 | 1.000 | 1.000 |
| min_judged_coverage@10 | 1.000 | 1.000 | 1.000 |
| failed_top10_total | 0 | 0 | 0 |

## document

| Metric | flask_current_k100 | previous_k50 | replay_k50 |
|---|---:|---:|---:|
| cases | 60 | 60 | 60 |
| cases_without_relevant | 2 | 2 | 2 |
| cases_without_relevant_binary | 4 | 4 | 4 |
| ndcg@10 | 0.845 | 0.847 | 0.847 |
| ndcg@10_binary | 0.824 | 0.828 | 0.834 |
| success@5 | 0.883 | 0.917 | 0.900 |
| success@10 | 0.933 | 0.933 | 0.933 |
| mrr@10 | 0.810 | 0.823 | 0.817 |
| precision@10 | 0.393 | 0.390 | 0.393 |
| precision@10_grade1 | 0.550 | 0.537 | 0.538 |
| precision@10_grade2 | 0.393 | 0.390 | 0.393 |
| mean_judged_coverage@10 | 1.000 | 1.000 | 1.000 |
| min_judged_coverage@10 | 1.000 | 1.000 | 1.000 |
| failed_top10_total | 0 | 0 | 0 |

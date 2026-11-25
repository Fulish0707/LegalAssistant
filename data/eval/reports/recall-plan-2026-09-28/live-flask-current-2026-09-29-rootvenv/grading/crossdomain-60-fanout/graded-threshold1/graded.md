# Graded retrieval comparison

- Judge: `ollama:qwen3:8b`, prompt `7a2b826ac9cb`, failed parses: 0, out-of-pool judgments ignored: 0
- Pool: pool_depth 10, settings_hash `live-k100-floor3-crossencoder-fp16-512`
- Headline relevance: grade ≥ 1 (success@k, mrr, precision@10, ndcg@10_binary)
- Corpora per run: `previous_k50` criminal-v1 + civil-v1 + administrative-v1 + ip-v1, `flask_current_k100` criminal-v1 + civil-v1 + administrative-v1 + ip-v1; filters None.

## chunk

| Metric | flask_current_k100 | previous_k50 |
|---|---:|---:|
| cases | 60 | 60 |
| cases_without_relevant_binary | 2 | 2 |
| ndcg@10_binary | 0.850 | 0.848 |
| success@5 | 0.950 | 0.950 |
| success@10 | 0.967 | 0.967 |
| mrr@10 | 0.862 | 0.866 |
| precision@10 | 0.592 | 0.580 |
| mean_judged_coverage@10 | 1.000 | 1.000 |
| min_judged_coverage@10 | 1.000 | 1.000 |
| failed_top10_total | 0 | 0 |

### Reference (graded)

| Metric | flask_current_k100 | previous_k50 |
|---|---:|---:|
| cases_without_relevant | 2 | 2 |
| ndcg@10 | 0.802 | 0.804 |
| precision@10_grade1 | 0.592 | 0.580 |
| precision@10_grade2 | 0.407 | 0.402 |

## document

| Metric | flask_current_k100 | previous_k50 |
|---|---:|---:|
| cases | 60 | 60 |
| cases_without_relevant_binary | 2 | 2 |
| ndcg@10_binary | 0.887 | 0.881 |
| success@5 | 0.950 | 0.950 |
| success@10 | 0.967 | 0.967 |
| mrr@10 | 0.903 | 0.907 |
| precision@10 | 0.550 | 0.537 |
| mean_judged_coverage@10 | 1.000 | 1.000 |
| min_judged_coverage@10 | 1.000 | 1.000 |
| failed_top10_total | 0 | 0 |

### Reference (graded)

| Metric | flask_current_k100 | previous_k50 |
|---|---:|---:|
| cases_without_relevant | 2 | 2 |
| ndcg@10 | 0.851 | 0.853 |
| precision@10_grade1 | 0.550 | 0.537 |
| precision@10_grade2 | 0.393 | 0.390 |

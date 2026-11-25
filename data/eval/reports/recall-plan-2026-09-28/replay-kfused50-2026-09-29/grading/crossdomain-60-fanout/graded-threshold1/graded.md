# Graded retrieval comparison

- Judge: `ollama:qwen3:8b`, prompt `7a2b826ac9cb`, failed parses: 0, out-of-pool judgments ignored: 0
- Pool: pool_depth 10, settings_hash `replay-k50-floor3-recorded-fp16-scores`
- Headline relevance: grade ≥ 1 (success@k, mrr, precision@10, ndcg@10_binary)
- Corpora per run: `previous_k50` criminal-v1 + civil-v1 + administrative-v1 + ip-v1, `flask_current_k100` criminal-v1 + civil-v1 + administrative-v1 + ip-v1; filters None.

## chunk

| Metric | flask_current_k100 | previous_k50 | replay_k50 |
|---|---:|---:|---:|
| cases | 60 | 60 | 60 |
| cases_without_relevant_binary | 2 | 2 | 2 |
| ndcg@10_binary | 0.843 | 0.841 | 0.841 |
| success@5 | 0.950 | 0.950 | 0.950 |
| success@10 | 0.967 | 0.967 | 0.967 |
| mrr@10 | 0.862 | 0.866 | 0.865 |
| precision@10 | 0.592 | 0.580 | 0.580 |
| mean_judged_coverage@10 | 1.000 | 1.000 | 1.000 |
| min_judged_coverage@10 | 1.000 | 1.000 | 1.000 |
| failed_top10_total | 0 | 0 | 0 |

### Reference (graded)

| Metric | flask_current_k100 | previous_k50 | replay_k50 |
|---|---:|---:|---:|
| cases_without_relevant | 2 | 2 | 2 |
| ndcg@10 | 0.795 | 0.798 | 0.804 |
| precision@10_grade1 | 0.592 | 0.580 | 0.580 |
| precision@10_grade2 | 0.407 | 0.402 | 0.403 |

## document

| Metric | flask_current_k100 | previous_k50 | replay_k50 |
|---|---:|---:|---:|
| cases | 60 | 60 | 60 |
| cases_without_relevant_binary | 2 | 2 | 2 |
| ndcg@10_binary | 0.881 | 0.875 | 0.873 |
| success@5 | 0.950 | 0.950 | 0.950 |
| success@10 | 0.967 | 0.967 | 0.967 |
| mrr@10 | 0.903 | 0.907 | 0.898 |
| precision@10 | 0.550 | 0.537 | 0.538 |
| mean_judged_coverage@10 | 1.000 | 1.000 | 1.000 |
| min_judged_coverage@10 | 1.000 | 1.000 | 1.000 |
| failed_top10_total | 0 | 0 | 0 |

### Reference (graded)

| Metric | flask_current_k100 | previous_k50 | replay_k50 |
|---|---:|---:|---:|
| cases_without_relevant | 2 | 2 | 2 |
| ndcg@10 | 0.845 | 0.847 | 0.847 |
| precision@10_grade1 | 0.550 | 0.537 | 0.538 |
| precision@10_grade2 | 0.393 | 0.390 | 0.393 |

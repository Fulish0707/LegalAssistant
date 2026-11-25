# Graded retrieval comparison

- Judge: `ollama:qwen3:8b`, prompt `7a2b826ac9cb`, failed parses: 0, out-of-pool judgments ignored: 0
- Pool: pool_depth 10, settings_hash `replay-k50-floor3-recorded-fp16-scores`
- Headline relevance: grade ≥ 2 (success@k, mrr, precision@10, ndcg@10_binary)
- Corpora per run: `previous_k50` criminal-v1 + civil-v1 + administrative-v1 + ip-v1, `flask_current_k100` criminal-v1 + civil-v1 + administrative-v1 + ip-v1; filters None.

## chunk

| Metric | flask_current_k100 | previous_k50 | replay_k50 |
|---|---:|---:|---:|
| cases | 200 | 200 | 200 |
| cases_without_relevant | 0 | 0 | 0 |
| cases_without_relevant_binary | 5 | 5 | 5 |
| ndcg@10 | 0.732 | 0.629 | 0.669 |
| ndcg@10_binary | 0.672 | 0.567 | 0.615 |
| success@5 | 0.925 | 0.900 | 0.910 |
| success@10 | 0.975 | 0.945 | 0.955 |
| mrr@10 | 0.710 | 0.690 | 0.714 |
| precision@10 | 0.544 | 0.449 | 0.490 |
| precision@10_grade1 | 0.790 | 0.690 | 0.718 |
| precision@10_grade2 | 0.544 | 0.449 | 0.490 |
| mean_judged_coverage@10 | 1.000 | 1.000 | 1.000 |
| min_judged_coverage@10 | 1.000 | 1.000 | 1.000 |
| failed_top10_total | 0 | 0 | 0 |

## document

| Metric | flask_current_k100 | previous_k50 | replay_k50 |
|---|---:|---:|---:|
| cases | 200 | 200 | 200 |
| cases_without_relevant | 0 | 0 | 0 |
| cases_without_relevant_binary | 5 | 5 | 5 |
| ndcg@10 | 0.744 | 0.637 | 0.680 |
| ndcg@10_binary | 0.684 | 0.579 | 0.629 |
| success@5 | 0.940 | 0.910 | 0.930 |
| success@10 | 0.975 | 0.945 | 0.955 |
| mrr@10 | 0.723 | 0.712 | 0.730 |
| precision@10 | 0.540 | 0.439 | 0.487 |
| precision@10_grade1 | 0.778 | 0.660 | 0.701 |
| precision@10_grade2 | 0.540 | 0.439 | 0.487 |
| mean_judged_coverage@10 | 1.000 | 1.000 | 1.000 |
| min_judged_coverage@10 | 1.000 | 1.000 | 1.000 |
| failed_top10_total | 0 | 0 | 0 |

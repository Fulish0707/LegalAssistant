# Graded retrieval comparison

- Judge: `ollama:qwen3:8b`, prompt `7a2b826ac9cb`, failed parses: 0, out-of-pool judgments ignored: 0
- Pool: pool_depth 10, settings_hash `live-k100-floor3-crossencoder-fp16-512`
- Headline relevance: grade ≥ 2 (success@k, mrr, precision@10, ndcg@10_binary)
- Corpora per run: `previous_k50` criminal-v1 + civil-v1 + administrative-v1 + ip-v1, `flask_current_k100` criminal-v1 + civil-v1 + administrative-v1 + ip-v1; filters None.

## chunk

| Metric | flask_current_k100 | previous_k50 |
|---|---:|---:|
| cases | 200 | 200 |
| cases_without_relevant | 0 | 0 |
| cases_without_relevant_binary | 5 | 5 |
| ndcg@10 | 0.748 | 0.641 |
| ndcg@10_binary | 0.689 | 0.580 |
| success@5 | 0.925 | 0.900 |
| success@10 | 0.975 | 0.945 |
| mrr@10 | 0.710 | 0.690 |
| precision@10 | 0.544 | 0.449 |
| precision@10_grade1 | 0.790 | 0.690 |
| precision@10_grade2 | 0.544 | 0.449 |
| mean_judged_coverage@10 | 1.000 | 1.000 |
| min_judged_coverage@10 | 1.000 | 1.000 |
| failed_top10_total | 0 | 0 |

## document

| Metric | flask_current_k100 | previous_k50 |
|---|---:|---:|
| cases | 200 | 200 |
| cases_without_relevant | 0 | 0 |
| cases_without_relevant_binary | 5 | 5 |
| ndcg@10 | 0.758 | 0.646 |
| ndcg@10_binary | 0.700 | 0.591 |
| success@5 | 0.940 | 0.910 |
| success@10 | 0.975 | 0.945 |
| mrr@10 | 0.721 | 0.710 |
| precision@10 | 0.539 | 0.438 |
| precision@10_grade1 | 0.777 | 0.659 |
| precision@10_grade2 | 0.539 | 0.438 |
| mean_judged_coverage@10 | 1.000 | 1.000 |
| min_judged_coverage@10 | 1.000 | 1.000 |
| failed_top10_total | 0 | 0 |

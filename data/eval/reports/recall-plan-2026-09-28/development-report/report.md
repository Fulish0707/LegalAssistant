# Development recall results

These are known-reference diagnostics on previously used questions. Relevance judgments and fresh confirmation are pending; no serving change is adopted.

Recall@10 below counts labeled reference documents found in the first 10 chunks. A domain's values are averaged over its questions; the headline gives each domain equal weight.

| Variant | Questions | Macro reference R@10 | Paired delta vs B0 | 95% CI |
| --- | ---: | ---: | ---: | --- |
| full_union | 40 | 0.450 | +0.000 | [-0.075, +0.075] |
| baseline | 100 | 0.455 | — | — |
| F0 | 100 | 0.446 | -0.009 | [-0.028, +0.000] |
| F1 | 100 | 0.455 | +0.000 | [+0.000, +0.000] |
| F2 | 100 | 0.446 | -0.009 | [-0.025, +0.000] |
| F3 | 100 | 0.455 | +0.000 | [+0.000, +0.000] |
| F4 | 100 | 0.455 | +0.000 | [+0.000, +0.000] |
| F5 | 100 | 0.455 | +0.000 | [+0.000, +0.000] |
| F6 | 100 | 0.455 | +0.000 | [+0.000, +0.000] |
| F7a | 100 | 0.455 | +0.000 | [+0.000, +0.000] |
| F7b | 100 | 0.444 | -0.011 | [-0.034, +0.000] |
| F7c | 100 | 0.418 | -0.037 | [-0.083, +0.001] |
| R0 | 100 | 0.399 | -0.056 | [-0.123, +0.013] |

Full-union results use only their fixed 40-question subset; their delta compares against B0 on those same questions. Do not compare that row's absolute score directly with the 100-question rows.

## Baseline stage losses

| Domain | Questions | Reference absent from union | Lost at fusion | Lost at top-10 reranking | Lost at document selection |
| --- | ---: | ---: | ---: | ---: | ---: |
| criminal | 29 | 2 | 0 | 4 | 1 |
| civil | 22 | 17 | 3 | 1 | 0 |
| administrative | 29 | 2 | 0 | 0 | 2 |
| ip | 20 | 15 | 4 | 0 | 0 |

Loss counts describe complete loss of known reference IDs, not loss of every relevant authority. Document selection does not establish passage survival: that needs validated supporting spans.

Fusion comparisons reuse fixed candidates and pointwise reranker scores. Their replay timings are not end-to-end serving latency. Confidence intervals are exploratory and are not adjusted for selecting among many variants.

Next gate: independently judge the shared top-10 pool before choosing a finalist. A fresh confirmation set remains unavailable.

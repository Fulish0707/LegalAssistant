# Candidate depth development comparison

Previously used questions and known source IDs only. No independently judged relevance or serving change.

| Depth per dense/sparse leg | Questions | Macro union coverage | Macro source R@10 | Paired R@10 delta vs 100 | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| 100 | 100 | 0.585 | 0.455 | — | — |
| 50 | 100 | 0.585 | 0.426 | -0.029 | [-0.065, +0.000] |
| 200 | 100 | 0.622 | 0.466 | +0.011 | [+0.000, +0.034] |

Each depth was a separate live retrieval at the same fused budget of 100; ANN rankings can change with requested depth.
Known-reference IDs are incomplete relevance labels; the result cannot select a serving variant.

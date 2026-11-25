# C5 query expansion development result

Known-source diagnostics on previously used questions; passage relevance remains unjudged.

| Retrieval | Questions | Macro union source coverage | Macro source R@10 |
| --- | ---: | ---: | ---: |
| baseline | 100 | 0.585 | 0.455 |
| depth50 | 100 | 0.585 | 0.426 |
| C5 | 100 | 0.585 | 0.435 |

Paired C5 minus B0 source R@10: -0.020 (exploratory 95% CI [-0.052, +0.000]).
Paired C5 minus depth-50 control: +0.009 (exploratory 95% CI [+0.000, +0.027]).

Accepted expansions: 52/100; fallback cases reproduce the depth-50 original-only control.
Repeated generation changed the accepted expansion on 2/20 queries.
Mean generation 562.5 ms; evaluator total p50/p95 2973.9/3345.8 ms. Capture serialization is included, so these are not serving latency estimates.

No winner is selected without independent passage labels. Novel citations trigger fallback and accepted terms may still be semantically poor.

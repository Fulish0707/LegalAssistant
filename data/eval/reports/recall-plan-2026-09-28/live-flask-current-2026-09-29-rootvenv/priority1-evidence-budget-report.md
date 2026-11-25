# Priority 1 follow-up — Evidence-budget counterfactual

- **Date:** 2026-09-29
- **Result type:** Development-only offline counterfactual; retrieval and serving unchanged
- **Scope:** Current captured top 10 for the existing 200 IP and 60 mixed-domain questions

## Method

Each grade-2 top-10 pair contributes its locally judged, quote-verified Qwen quote. The relevance judge verifier normalization is used: whitespace is collapsed and surrounding quote marks and ellipses are stripped. The stricter prior own-chunk count is also reconciled below. The analysis replays Flask's current formatter and confirms its reconstructed ordered evidence equals the captured evidence for every case. It then compares the baseline with alternatives under the same cap of five documents and 1,200 characters per document (at most 6,000 characters per question).

The document-choice and excerpt-placement variants are **label-aware upper bounds**: they use the same provisional quotes they score to pick documents or excerpt locations. They show where the current budget can lose those quotes; they do not estimate how a real selector would perform. The sentence-window variant keeps the current prefix as a fallback. The character-window variant allows any contiguous excerpt of the same length.

## Baseline reconciliation

- IP: the prior whitespace-only own-chunk count reproduces at 457/457. Applying the judge verifier's edge stripping adds 2; counting matching quote text in a sibling chunk of the same displayed document adds 1 more, for 460 source-document quotes.
- Mixed domains: the prior whitespace-only own-chunk count reproduces at 116/116. Applying the judge verifier's edge stripping adds 1; counting matching quote text in a sibling chunk of the same displayed document adds 2 more, for 119 source-document quotes.

## Quote survival

Counts are shown as visible source-document quotes / all verified grade-2 quotes. Case counts use questions with at least one verified grade-2 quote in the top 10 as the denominator. Cases newly visible are the number with no baseline quote that gain at least one under the variant.

| Dataset | Evidence view | Quote pairs | Pair retention | Cases with a visible quote | Case retention | New cases vs baseline | Pair change vs baseline |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IP (200) | Current top 5 + prefix | 460/1089 | 42.2% | 171/195 | 87.7% | 0 | +0 |
| IP (200) | Same top 5 + sentence-window oracle | 570/1089 | 52.3% | 187/195 | 95.9% | 16 | +110 |
| IP (200) | Same top 5 + character-window oracle | 572/1089 | 52.5% | 187/195 | 95.9% | 16 | +112 |
| IP (200) | Best 5 + prefix oracle | 701/1089 | 64.4% | 187/195 | 95.9% | 16 | +241 |
| IP (200) | Best 5 + sentence-window oracle | 823/1089 | 75.6% | 195/195 | 100.0% | 24 | +363 |
| IP (200) | Best 5 + character-window oracle | 824/1089 | 75.7% | 195/195 | 100.0% | 24 | +364 |
| Mixed domains (60) | Current top 5 + prefix | 119/244 | 48.8% | 48/56 | 85.7% | 0 | +0 |
| Mixed domains (60) | Same top 5 + sentence-window oracle | 148/244 | 60.7% | 53/56 | 94.6% | 5 | +29 |
| Mixed domains (60) | Same top 5 + character-window oracle | 150/244 | 61.5% | 53/56 | 94.6% | 5 | +31 |
| Mixed domains (60) | Best 5 + prefix oracle | 154/244 | 63.1% | 50/56 | 89.3% | 2 | +35 |
| Mixed domains (60) | Best 5 + sentence-window oracle | 188/244 | 77.0% | 56/56 | 100.0% | 8 | +69 |
| Mixed domains (60) | Best 5 + character-window oracle | 190/244 | 77.9% | 56/56 | 100.0% | 8 | +71 |

## Interpretation

- The same-top-five rows isolate excerpt placement. Compare their case gains as well as pair gains; a high pair-count gain can come from many quotes on questions that already had some visible support.
- The best-five rows isolate document selection when using prefixes, then show the combined document-plus-excerpt ceiling. These are hindsight oracles, not candidate production behavior.
- The gap between sentence-window and character-window ceilings indicates how much of the possible gain depends on breaking sentence boundaries or selecting inside a long sentence.
- These are quote-retention measurements, not answer-quality measurements or classic Recall@k. Human support-span labels and independent calibration remain unavailable.

## Provenance

Judge: `ollama:qwen3:8b`, prompt `7a2b826ac9cb93494241c98ccc829755d2af0ee37ebd8e3178cdbd798ecb596c`, `think=true`. Independent human calibration was unavailable. Input hashes and the reproducible analysis script are recorded in `priority1-evidence-budget-analysis.json`.

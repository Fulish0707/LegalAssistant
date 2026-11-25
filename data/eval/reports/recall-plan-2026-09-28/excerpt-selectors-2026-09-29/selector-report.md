# Priority 1 follow-up: label-free excerpt selectors against the evidence-budget oracles

- **Date:** 2026-09-29
- **Result type:** Development-only offline counterfactual; retrieval, ranking and serving unchanged
- **Design and pass bar:** `design.md`, written before any selector was scored

## Quote survival

Same top-10 evidence, same formatter and quote matching as `priority1-evidence-budget-report.md`. Every row shows at most five documents and 1,200 characters per document. Oracle rows are copied from that report. Gained and lost count questions with a visible verified grade-2 quote against the current prefix; the gap closed is net new questions over the oracle's new questions.

| Dataset | Evidence view | Quote pairs | Cases with a visible quote | Cases gained / lost | Net cases | Share of oracle case gap closed | Pairs gained / lost | Shown documents holding a quote (of 5) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IP (200) | Current top 5 + prefix | 460/1089 | 171/195 | +0 / -0 | +0 |  | +0 / -0 | 2.35 |
| IP (200) | Same top 5 + cross-encoder window | 449/1089 | 171/195 | +5 / -5 | +0 | 0% | +23 / -34 | 2.30 |
| IP (200) | Same top 5 + lexical window | 389/1089 | 167/195 | +9 / -13 | -4 | -25% | +35 / -106 | 1.99 |
| IP (200) | Top 5 by cross-encoder window + prefix | 446/1089 | 169/195 | +8 / -10 | -2 | -12% | +101 / -115 | 2.28 |
| IP (200) | Top 5 by cross-encoder window + its window | 454/1089 | 173/195 | +10 / -8 | +2 | 8% | +133 / -139 | 2.32 |
| IP (200) | Top 5 by lexical window + its window | 370/1089 | 166/195 | +11 / -16 | -5 | -21% | +160 / -250 | 1.90 |
| IP (200) | Oracle: same top 5 + character window | 572/1089 | 187/195 | | +16 | 100% | | |
| IP (200) | Oracle: best 5 + prefix | 701/1089 | 187/195 | | +16 | 100% | | |
| IP (200) | Oracle: best 5 + character window | 824/1089 | 195/195 | | +24 | 100% | | |
| Mixed domains (60) | Current top 5 + prefix | 119/244 | 48/56 | +0 / -0 | +0 |  | +0 / -0 | 2.11 |
| Mixed domains (60) | Same top 5 + cross-encoder window | 124/244 | 50/56 | +5 / -3 | +2 | 40% | +14 / -9 | 2.20 |
| Mixed domains (60) | Same top 5 + lexical window | 116/244 | 48/56 | +4 / -4 | +0 | 0% | +15 / -18 | 2.05 |
| Mixed domains (60) | Top 5 by cross-encoder window + prefix | 116/244 | 48/56 | +0 / -0 | +0 | 0% | +16 / -19 | 2.05 |
| Mixed domains (60) | Top 5 by cross-encoder window + its window | 128/244 | 51/56 | +6 / -3 | +3 | 38% | +35 / -26 | 2.27 |
| Mixed domains (60) | Top 5 by lexical window + its window | 110/244 | 47/56 | +5 / -6 | -1 | -12% | +39 / -48 | 1.95 |
| Mixed domains (60) | Oracle: same top 5 + character window | 150/244 | 53/56 | | +5 | 100% | | |
| Mixed domains (60) | Oracle: best 5 + prefix | 154/244 | 50/56 | | +2 | 100% | | |
| Mixed domains (60) | Oracle: best 5 + character window | 190/244 | 56/56 | | +8 | 100% | | |

## Pass bar (fixed in design.md)

| Selector | Dataset | Net new cases | Needed | Cases lost (max 1) | Net pairs | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Same top 5 + cross-encoder window | IP (200) | +0 | 8 | 5 | -11 | miss |
| Same top 5 + cross-encoder window | Mixed domains (60) | +2 | 3 | 3 | +5 | miss |
| **Same top 5 + cross-encoder window** | **both** | | | | | **MISS** |
| Top 5 by cross-encoder window + its window | IP (200) | +2 | 12 | 8 | -6 | miss |
| Top 5 by cross-encoder window + its window | Mixed domains (60) | +3 | 4 | 3 | +9 | miss |
| **Top 5 by cross-encoder window + its window** | **both** | | | | | **MISS** |

## Does the score find the quote?

Share of verified quotes that sit inside the highest-scoring window of their document (before it is widened to 1,200 characters), inside one of the three highest, and inside the first window. Chance is the average share of a document's windows that contain the quote.

| Dataset | Quotes located | Cross-encoder top 1 | Cross-encoder top 3 | Lexical top 1 | Lexical top 3 | First window | Chance (top 1) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IP (200) | 1082 | 69.1% | 89.1% | 58.4% | 85.1% | 64.5% | 38.9% |
| Mixed domains (60) | 244 | 72.5% | 89.3% | 65.2% | 85.2% | 55.7% | 32.2% |

## Serving cost

Measured on the RTX 4060 laptop GPU, fp16, batch 16: 15.4 ms per (query, window) pair. Scoring all windows of the ten documents takes 66 pairs, about 1.01 s per question (measured); scoring only the first five documents takes 36 pairs, about 0.55 s (estimated from the per-pair time). For scale, `k_fused` 100 against 50 cost about 0.7 s.

## Provenance

Judge: `ollama:qwen3:8b`, prompt `7a2b826ac9cb93494241c98ccc829755d2af0ee37ebd8e3178cdbd798ecb596c`, `think=true`; no new judgments. Independent human calibration was unavailable. Each pair has one Qwen-chosen quote, so a selector that shows a different passage that also answers the question scores as a miss. Input hashes are in `selector-analysis.json`.

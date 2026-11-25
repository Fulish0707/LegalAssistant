# GPT-6 provisional AI-rater comparison

Date: 2026-09-28  
Calibration sample: `calibration-100-final/` (100 frozen query–passage pairs)  
Comparator: `ollama:qwen3:8b` (think enabled)  
Reference rater: GPT-6 Codex, AI

## Overall agreement

The reviewers matched all 100 pair IDs; no Qwen grades were missing. For relevance at grade 1 or 2, observed agreement was 0.890 and Cohen’s κ was 0.719. This exceeds the predeclared 0.60 threshold for this AI-to-AI comparison.

| Measure | Result |
|---|---:|
| Exact agreement across grades 0/1/2 | 0.660 |
| Linear weighted Cohen’s κ across grades 0/1/2 | 0.563 |
| Cohen’s κ for grade ≥ 1 | 0.719 |
| Cohen’s κ for grade 2 vs. grades 0–1 | 0.442 |
| GPT-6 grade counts (0 / 1 / 2) | 29 / 22 / 49 |

Precision (0.895) and recall (0.958) are calculated relative to the GPT-6 AI labels. They are agreement statistics, not measurements against human ground truth.

During schema inspection, the Qwen outputs for the first three rows were visible before those GPT-6 labels were assigned, so the 100-pair comparison is not fully blind. Excluding those three pairs gives a 97-pair sensitivity result: exact agreement 0.649, weighted κ 0.543, relevant-label κ 0.701, and grade-2-vs-lower κ 0.424. The relevant-label value remains above 0.60, but it is still AI-to-AI evidence only. Full details are in `agreement-run/sensitivity-97.json`.

## Domain diagnostics

| Domain | Pairs | Exact agreement | κ, grade ≥ 1 | κ, grade 2 vs. lower |
|---|---:|---:|---:|---:|
| Criminal | 25 | 0.600 | 0.531 | 0.299 |
| Civil | 25 | 0.720 | 1.000 | 0.364 |
| Administrative | 25 | 0.600 | 0.525 | 0.590 |
| IP | 25 | 0.720 | 1.000 | 0.345 |

Per-domain values are diagnostic only. Each domain has 25 pairs; civil and IP also have high positive-label prevalence, so their κ values should not be read as broad proof of judge reliability. Grade-2 agreement is weaker than binary relevant/not-relevant agreement.

## Interpretation and remaining work

The threshold is met only for this comparison between two AI raters. It does not supply independent human labels, calibrate the Qwen judge, validate semantic Recall@k, or establish retrieval quality. The Qwen run remains `uncalibrated_complete`, and the shared judging pool remains unjudged.

Independent review of the 100 calibration pairs is still required before calibrating the judge. The separate 200-question confirmation remains pending because no unused questions are available. No serving or retrieval configuration changed.

## Files

- `reviewed-calibration.csv`: GPT-6 AI labels; the frozen source CSV is unchanged.
- `reviewer-manifest.json`: reviewer type, date, rubric, and file hashes.
- `agreement-run/agreement.json`: complete overall and per-domain agreement output from the project evaluator.

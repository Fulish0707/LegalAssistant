# Excerpt-selector test (2026-09-29)

Result: both pre-registered selectors (cross-encoder best window, and best window plus document re-selection) missed the
pass bar on both development sets, and a lexical selector was worse than the current prefix. Not adopted; no serving change.
Headline tables are in `selector-report.md`; the interpretation is in `docs/retrieval-diagnostics.md`
(section "Label-free excerpt selectors").

Order of work (run from the worktree root with `PYTHONPATH=src /home/layke/Documents/MyLLMAgent/.venv/bin/python`):

1. `design.md` was written first: variants, metrics, pass bar. The erratum at its end was added after the results.
2. `windows.py` builds label-free windows; `score_windows.py` scores them with the serving cross-encoder (GPU, about 4 minutes;
   writes `window-scores.jsonl` and `score-run.json`, including the query-equality and recorded-score checks).
3. `evaluate_selectors.py` (CPU) first requires the harness to reproduce the evidence-budget report's baseline and oracle rows,
   then writes `selector-analysis.json`, `selector-cases.json` and `selector-report.md`.

Post-hoc and exploratory, not part of the pre-registered design: `posthoc_margin.py` (keep the prefix unless a window scores
clearly higher; `posthoc-margin-sweep.txt`) and `posthoc_lost_cases.py` (four IP questions where the cross-encoder window lost a
quote the prefix showed; `posthoc-lost-cases.txt`).

Limits: each pair has one Qwen-selected quote, so a selector that shows another passage that also answers the question is
scored as a miss; judge calibration and human labels do not exist; both sets are the development sets the oracles were built on.

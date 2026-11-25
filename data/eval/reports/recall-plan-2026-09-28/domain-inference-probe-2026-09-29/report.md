# Query analyzer domain probe — 2026-09-29

- Cases: 60 existing mixed-domain development questions, 20 each for criminal, civil, and administrative. No new or confirmation questions were created.
- Analyzer: `ollama:qwen3:4b`, temperature 0; single-turn input with no chat history.
- Domain label mapping: 민사 → civil, 형사 → criminal, 행정 → administrative. Other or missing labels remain unmapped and would need to fall back to fan-out.
- Mapped coverage: 49/60 (0.817).
- Correct routes over all cases (unmapped counted as not routed correctly): 29/60 (0.483).
- Accuracy among mapped cases: 29/49 (0.592) if any were mapped.
- Per-domain counts and the confusion matrix are in `summary.json`; case-level predictions are in `predictions.jsonl`.

This measures classification behavior only. It does not establish that domain-based routing improves graded retrieval quality. The prompt's supported domain labels and mapping are recorded in the manifest.

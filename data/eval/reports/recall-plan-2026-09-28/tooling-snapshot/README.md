# Archived recall-stage harness (2026-09-29)

Copied on 2026-11-25 from the uncommitted work of the removed `codex/rag-recall-evaluation`
worktree, so that the results in `docs/retrieval-diagnostics.md` (Priority 1, `k_fused=50`
replay, excerpt selectors, routing headroom) keep the code that produced them.

Not maintained and not on the import path. Main later moved
`legal_rag.evaluation.{calibration,judgment_store,relevance_judge}` to
`legal_rag.evaluation.relevance.*`, so `scripts/eval_recall_stages.py` needs those imports
changed before it runs. The tests here are renamed `*.py.txt` so pytest does not collect them.

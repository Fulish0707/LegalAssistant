# Smaller structural retrieval chunks and expanded answer context

Approved in conversation: retrieve and rank smaller structural chunks, then add
surrounding passages for generation. Initial civil sizing is target 800 characters,
hard maximum 1000, using the existing structural parser and sentence fallback.

Create `civil-struct800-v1` from the unchanged civil-v1 documents. Existing corpora
and collections remain the comparison baseline. Build a separate dense/BM25 index;
no automatic deployment or replacement of the running API profile.

Expansion is restricted to civil-struct800-v1 in the pilot profile, avoiding
unindexed neighbor scans on old domain collections. Expansion reads at most one preceding and one following chunk of the same document
and corpus version after reranking. Anchors take priority; neighbors are deduplicated,
are separately cited and have their own legal headers, and are not represented as
independently reranked hits. Total rendered evidence is bounded at 7500 characters.
Generic defaults leave expansion disabled; a separate structural-civil profile
enables it and selects the new civil corpus alongside the existing other domains.

Qdrant stores document_id and order already. Add payload indexes for these fields
on new collections, and use bounded filtered scroll to obtain neighbors. Route
lookups only to the store for the hit's corpus version, never across versions.

Acceptance: structural corpus is prepared; index command is runnable; runtime
wiring supports version-scoped expansion and accurate citations; docs describe
comparison and rollout. Static checks and code review only in this implementation
turn: developer instruction prohibits adding/running tests without a user request.
Full dense indexing, live retrieval evaluation and measured quality improvement
are pending the user's indexing/evaluation run.

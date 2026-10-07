# Structural Context Implementation Plan

**Goal:** Rank smaller structural chunks and expand answer evidence with bounded neighboring passages.
**Architecture:** Existing structural chunker creates a separate civil version. The hybrid retriever supplies version-scoped Qdrant neighbor lookups to the context builder after ranking.
**Tech stack:** Python, Pydantic settings, Qdrant, existing structural parser.

- [x] Add bounded document/order lookup and payload indexes in `retrieval/store.py`.
- [x] Route neighbor requests to the matching corpus store in `retrieval/hybrid.py`.
- [x] Add optional neighbor radius and rendered-context budget settings.
- [x] Expand anchors first, deduplicate neighbor chunks and preserve separate citations in `rag/context.py`; wire via orchestrator.
- [x] Add a separate structural-civil profile with 7500-character rendered evidence budget and radius one.
- [x] Prepare civil-struct800-v1 with `scripts/rechunk_corpus.py --corpus-version civil-v1 --output-version civil-struct800-v1 --target-chars 800 --max-chars 1000 --merge-threshold 400`.
- [x] Document the separate index build and comparable baseline/variant evaluation.
- [x] Check syntax, CLI settings and diff; obtain static review. No tests added or run unless requested.

Protected scope: existing evaluation artifacts, original documents, old corpus/index,
.env secrets, and the other agent's statute-fetching work. Do not modify execution
state owned by a different mission. Do not auto-commit or switch the active API.

## Evidence

- Prepared all 75,624 documents: 661,901 chunks, median697 chars, maximum1000.
- Source/copy document SHA256 match: 6e17ba5a16ba2292810305f31f4e4671856e40de22375a50fd00fc8317f8bf70.
- Python syntax, CLI help, settings resolution and diff whitespace checks succeeded.
- Installed Qdrant scroll signature supports scroll_filter.
- Static review plus followup: no important defects remaining.
- No tests added/run, no full indexing, and no live neighbor retrieval or quality evaluation.
- User indexing and paired evaluation commands are documented in docs/structural-civil.md.

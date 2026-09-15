# External processed corpus implementation plan

**Goal:** Import the supplied IP corpus without changing source data, then verify retrieval and attempt real indexing.
**Architecture:** Adapt external chunks into existing retrieval contracts; preserve flat source documents via an explicit external-document reader. Publish a validated corpus atomically and reuse the hybrid indexer.
**Tech stack:** Python, Pydantic, JSONL, pytest, embedded Qdrant, Ollama.

- [x] Add tests in tests/legal_rag/test_import_processed.py for schema adaptation, original metadata and IDs, document lookup, rejection of duplicate/empty/orphaned records, atomic failure, sample isolation, and filtered Qdrant retrieval. Run `/tmp/legal-rag-import-venv/bin/python -m pytest tests/legal_rag/test_import_processed.py -q` and observe missing-feature failures.
- [x] Extend schemas/documents.py and schemas/chunks.py for external types and preserved JSON source fields. Add ingestion/import_processed.py and scripts/import_processed.py. Extend ingestion/corpus.py to read imported documents explicitly. Keep QA excluded and source files read-only.
- [x] Run the focused tests, then the existing legal_rag regression suite. Validate all 76,142 documents and 166,519 chunks with the CLI and import a complete-document sample into a separate version.
- [x] Run real embedding preflight, index/search the sample, and attempt the full isolated index. Record any service/model blocker and the exact continuation commands.
- [x] Update docs/legal-rag-backend.md, review the diff, record test and data evidence, and leave changes uncommitted.

Full import succeeded; real 32-chunk indexing and retrieval verified. Full indexing job is running; see .execute-protocol/progress.md.

# Import preprocessed legal corpora

The user approved an adapter, additional document types, validation, a small indexing trial, and a full IP corpus load. Preserve the original files and existing indexes. Keep the existing legal_rag package and hybrid retrieval architecture.

Read only explicitly supplied documents.jsonl and chunks.jsonl; never read QA files. Validate IDs, text, document references, domain/type consistency, chunk positions and declared counts. Reject ambiguous duplicates rather than silently dropping or overwriting them. Preserve original identifiers and source metadata in a namespaced JSON field; retain all external document type labels as explicit enum members. Preserve flat imported documents through a separate corpus reader path, because inventing article/decision structure would misrepresent source data.

Write canonical chunks and flat source documents with original headers preserved into a new corpus directory using a staging directory and publish only after validation succeeds. Add an external document adapter to Corpus so document lookup remains available. Produce a standard manifest and validation report with source hashes. Documents without any chunks are preserved and reported, while incomplete declared chunk sets fail. A sample selects complete documents; it must not truncate validation of source input. A separate corpus version isolates the sample and the full index. Existing collection naming remains model/version-specific.

Use existing BM25 and dense indexing. Validate real data and run an embedded-Qdrant retrieval test with a fake embedder before attempting Ollama. No claim of semantic quality or QA leakage certification follows from structural checks. Do not commit automatically.

Review refinements: reject duplicate JSON keys at every nesting level; preserve generic dates without guessing their legal meaning. Ollama embedding requests disable silent truncation so overlength input fails visibly.

Full-data scaling observation: the old indexer retained all Chunk objects and all token lists; its process reached 22 GB RSS and terminated before the first indexed batch. The implementation now streams corpus BM25 statistics in one pass, rereads chunks in bounded batches for embedding/upsert, and closes the store on errors. BM25 vocabulary statistics and embedded Qdrant storage still scale with corpus size.

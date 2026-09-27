# Korean Legal RAG Backend — Developer Guide

The `legal_rag` package under `src/` is the backend described in
`docs/superpowers/specs/2026-09-07-korean-legal-rag-backend-design.md`. It is independent
of the Flask app at the repository root. This guide covers install, the data workflow,
the chat interfaces, configuration, and how to extend it.

## Install

Python 3.11 is managed by `uv`. Ollama must be running for the real embedder and LLM.

```bash
uv python install 3.11
uv sync --group dev
ollama pull qwen3:4b
ollama pull bge-m3
```

`bge-m3` is the default embedding model because it handles Korean well. Any Ollama
embedding model works; set `LEGAL_RAG__EMBEDDING__MODEL=nomic-embed-text` to use one
you already have. Switching embedding models creates a new Qdrant collection, so rebuild
the index afterwards.

Optional extras:

```bash
uv sync --extra kiwi     # morpheme tokenizer for the BM25 leg (sparse.tokenizer: kiwi)
uv sync --extra rerank   # cross-encoder reranker (rerank.backend: cross_encoder)
```

## Tests

```bash
uv run pytest tests/legal_rag -q                      # offline, no network, ~2 s
LEGAL_RAG_LIVE=1 uv run pytest tests/legal_rag/live -q  # against local Ollama
```

The Flask app keeps its own suite under `tests/` and runs with `unittest` as before.

## Data workflow

Raw data is immutable and lives under `data/raw/<dataset>/`. The dataset folder name
becomes `source.dataset` on every document and part of every document id.

```bash
# 1. raw → processed corpus (documents.jsonl, chunks.jsonl, manifest.json)
uv run python scripts/ingest.py --raw-dir data/raw --corpus-version v1

# 2. processed corpus → BM25 statistics + dense vectors → embedded Qdrant
uv run python scripts/build_index.py --corpus-version v1

# 3. debug retrieval
uv run python scripts/search.py "민법 제750조" --corpus-version v1 --k 5

# 4. retrieval metrics (Recall@k, Precision@k, MRR, nDCG@k)
uv run python scripts/eval_retrieval.py --dataset data/eval/retrieval.jsonl --corpus-version v1
uv run python scripts/eval_retrieval.py --dataset data/eval/retrieval.jsonl --corpus-version v1 --leg sparse
```

Rebuilding a corpus from the same raw files and the same `PIPELINE_VERSION`
(`src/legal_rag/ingestion/__init__.py`) produces byte-identical `documents.jsonl` and
`chunks.jsonl`. `manifest.json` records every raw file's hash, the config hash, counts per
stage, and every dropped record with its reason.

Recognized raw layouts (`src/legal_rag/ingestion/detect.py`):

| Schema | Fingerprint keys | Normalizer |
| --- | --- | --- |
| `aihub_judgment_v1` | `info`, `facts`, `dcss`, `close` | `ingestion/normalize/aihub_judgment.py` |
| `statute_json_v1` | `law_name`, `articles` | `ingestion/normalize/statute_json.py` |
| `admin_interpretation_v1` | `authority`, `reference_number`, `question`, `interpretation` | `ingestion/normalize/admin_interpretation.py` |

Files that match no fingerprint are counted as `unrecognized` in the manifest and skipped;
the run does not fail.

## Chat

```bash
uv run python scripts/chat.py --corpus-version v1 --trace
```

Inside the loop: `/sources` prints the evidence blocks of the last answer, `/trace`
toggles stage timings and policy decisions, `/quit` exits.

Each turn runs: query analysis (deterministic reference extraction plus an LLM pass) →
clarification check → query rewriting → hybrid retrieval (dense + BM25, RRF fusion,
reranker) → evidence blocks `S1..Sn` → structured `LegalAnswer` → deterministic
verification → policy (accept, regenerate once, strip, or reject).

## HTTP API

```bash
uv run python scripts/serve.py --corpus-version v1 --port 8000
```

| Route | Body | Returns |
| --- | --- | --- |
| `GET /health` | – | corpus version, collection, chunk count, embedder, LLM |
| `POST /v1/search` | `{"query", "k", "filters"}` | ranked chunks with scores |
| `POST /v1/chat` | `{"message", "conversation_id", "include_trace"}` | `ChatResponse` (`kind` = `answer`, `clarification`, or `rejected`) |

Conversation history is stored in `data/indexes/<corpus_version>/conversations.db`. Only
the last few turns feed query analysis; evidence is rebuilt each turn and never stored as
history text.

## Configuration

Settings load in this order, later wins: `configs/default.yaml` → `configs/<profile>.yaml`
(profile from `LEGAL_RAG_PROFILE`, default `local`) → CLI options → environment variables
`LEGAL_RAG__SECTION__KEY`. Secrets such as `qdrant.api_key` are environment-only and are
rejected if they appear in YAML.

Frequently changed keys:

| Key | Default | Notes |
| --- | --- | --- |
| `corpus.version` | `v1` | Names the processed corpus, index directory, and collection |
| `embedding.backend` / `embedding.model` | `ollama` / `bge-m3` | `fake` is for tests |
| `sparse.tokenizer` | `regex` | `kiwi` needs the extra |
| `qdrant.mode` | `embedded` | `server` uses `qdrant.url` and `LEGAL_RAG__QDRANT__API_KEY` |
| `retrieval.k_dense`, `k_sparse`, `k_fused`, `k_final` | 20, 20, 20, 5 | Depth per leg, after fusion, and into the answer |
| `rerank.backend` | `noop` | `cross_encoder` needs the extra |
| `rerank.device` / `rerank.dtype` | `cpu` / `float32` | `local.yaml` uses `cuda` / `float16`, see below |
| `rerank.local_files_only` | `false` | Load the cached model without Hub checks (1.9 s instead of 5.6 s); `local.yaml` sets it |
| `api.warmup` | `false` | Run one search in the background at startup; `/health` reports `warm`. `local.yaml` sets it: first search after warm-up 2.3 s instead of ~18 s cold |
| `llm.backend` / `llm.model` | `ollama` / `qwen3:4b` | `transformers` arrives with V0.4 |
| `rag.allow_clarification` | `true` | Ask for facts instead of answering when the analyzer flags them |
| `rag.llm_rewrite` | `false` | Add LLM-generated search phrasings |
| `rag.max_regenerations` | `1` | Regeneration attempts after a failed verification |
| `trace.path` / `trace.include_text` | unset / `false` | JSONL audit log; user text is excluded unless enabled |

Embedded Qdrant allows one client per storage path. Build the index before starting the
API, and use `qdrant.mode: server` when several processes must share an index.

## Extending

**Add a raw layout.** Add a fingerprint to `ingestion/detect.py`, a normalizer under
`ingestion/normalize/` that returns a canonical document, register it in
`ingestion/normalize/__init__.py`, and add a fixture under `tests/legal_rag/fixtures/raw/`.

**Swap a backend.** Every stage is a `Protocol`: `Embedder`, `Tokenizer`, `Retriever`,
`Fuser`, `Reranker`, `ChatModel`, `ConversationStore`. Implement the Protocol and add a
branch in `src/legal_rag/factory.py`.

**Change verification.** Checks live in `verification/verifier.py`; what happens to a
failed report lives in `verification/policy.py`; `verification/backfill.py` restores
citations for authorities the prose names and the evidence contains.

## Roadmap

V0.4 adds the `training/` package (RAG-aware SFT dataset, QLoRA config, adapter registry)
and a Transformers `ChatModel` so experiments A–D (base vs tuned, with and without RAG) are
pure configuration. V0.5 adds temporal validity and claim-level grounding checks.

## Import externally preprocessed JSONL

Use `scripts/import_processed.py` for data produced outside this repository. It reads
only `documents.jsonl` and `chunks.jsonl` from the supplied directory. QA files are
never read or indexed. Original files remain unchanged.

```bash
# Validate the entire source without creating a corpus.
uv run python scripts/import_processed.py \
  --input-dir /home/layke/Documents/legal-rag/data/processed/intellectual_property \
  --corpus-version ip-v1 --validate-only

# Import a sample containing all chunks of the first two documents.
uv run python scripts/import_processed.py \
  --input-dir /home/layke/Documents/legal-rag/data/processed/intellectual_property \
  --corpus-version ip-sample-v1 --max-documents 2
ollama pull bge-m3
uv run python scripts/build_index.py --corpus-version ip-sample-v1
uv run python scripts/search.py '상표권 침해' --corpus-version ip-sample-v1 --k 5

# Import and index the full corpus in a separate version.
uv run python scripts/import_processed.py \
  --input-dir /home/layke/Documents/legal-rag/data/processed/intellectual_property \
  --corpus-version ip-v1
uv run python scripts/build_index.py --corpus-version ip-v1
uv run python scripts/search.py '특허권 침해' --corpus-version ip-v1 --k 5
```

The importer requires `document_id`, `document_type`, `legal_domain`, and nonempty
`text` on documents. Chunks additionally require `chunk_id`, `chunk_index`, and
`total_chunks`. It preserves chunk IDs as `id`, copies chunk position into `order`,
and stores the original record (excluding text, which stays in `text`) under
`metadata.original`. The original metadata can therefore contain nested JSON.
External document types (`judgment`, `legal_interpretation`, `decision`,
`patent_trial_decision`, `constitutional_decision`) remain distinct; native backend
types continue to work. Imported flat documents carry an explicit format marker;
no article hierarchy or decision sections are invented.

Duplicate IDs, empty text, missing document references, mismatched domains/types,
and incomplete or duplicate chunk positions fail the import. Documents without any
chunks are preserved and listed in `validation_report.json`; they are not searchable.
Source hashes are recorded in the standard `manifest.json` and validation report.
A sample still validates the entire source before publishing its selected records.

Output goes to `data/processed/<corpus-version>/`. Existing output directories are
never overwritten: choose a new version to import changed data. Publication occurs
only after validation finishes. Collections retain the backend's model/version-based
names; this importer does not switch to a fixed `legal_chunks` collection.

**Indexing remains a full rebuild:** `build_index.py` recreates that version/model's
collection. Use a new corpus version to preserve an existing index. Embedded Qdrant
requires exclusive access; stop search/API processes using the same index first.
Full embedding of a large corpus can take substantial time. `fake` embeddings are
only suitable for pipeline tests, not semantic search.

Structural validation does not prove that earlier preprocessing excluded QA gold
answers. That requires reviewing the original preprocessing and QA/source provenance.

The Ollama embedder sets `truncate: false`: overlength chunks cause an error instead
of silently losing their ending. Generic source `date` values remain in
`metadata.original`; only explicitly named `decision_date` and `effective_date`
metadata populate those legal date filters.

Index preparation uses two streaming passes: first fit corpus BM25 frequencies,
then embed and upsert batches. It does not retain every chunk or token list in RAM.
Embedded Qdrant still uses memory for its stored vectors and payloads; streaming
preparation does not make total database memory constant.

## Document type and citation kind

`document_type` is the fine-grained retrieval facet. Six values appear across the four
AI Hub domains: `statute`, `judgment`, `legal_interpretation`, `constitutional_decision`,
`decision`, and `patent_trial_decision`. `court_decision` and `admin_interpretation`
remain the native normalizers' spellings for structured sources.

Evidence rendering and citation verification need only a coarse reading, so
`citation_kind(document_type)` in `schemas/answer.py` derives a `CitationKind` of
`statute`, `case`, or `interpretation`. Every adjudicative type maps to `case`. That
mapping is the single source of truth for `rag/context.py`, `verification/backfill.py`,
and the importer's `court` field.

Do not test `document_type` for equality against one spelling. Doing so is what caused
imported judgments to render as interpretations with empty authority fields.

Retrieval can filter by either axis. `RetrievalFilters.citation_kind` expands to a
`MatchAny` over that kind's member document types, so a category filter runs server-side
with no extra payload field. An explicit `document_types` list wins, letting a caller
narrow within a category.

```python
RetrievalFilters(citation_kind=CitationKind.case)                  # any adjudicative source
RetrievalFilters(document_types=[DocumentType.patent_trial_decision])  # just this one
```

A new `DocumentType` member must be added to `_CITATION_KINDS`;
`test_citation_kind_covers_every_document_type` fails otherwise.

## Retrieval evaluation datasets

`scripts/build_eval_set.py` converts an external `qa.jsonl` into the `RetrievalExample`
rows `eval_retrieval.py` consumes. Only the question and its `document_id` cross over.
The gold answer never enters the dataset — `test_gold_answer_never_reaches_the_dataset`
fails if it does.

```bash
uv run python scripts/build_eval_set.py \
  --qa-file /home/layke/Documents/legal-rag/data/processed/criminal/qa.jsonl \
  --corpus-version criminal-v1

uv run python scripts/eval_retrieval.py \
  --dataset data/eval/criminal-v1.jsonl --corpus-version criminal-v1
```

A question whose `document_id` is absent from the corpus is dropped and counted, because
it can never be retrieved and would depress Recall@k for a data-matching reason rather
than a retrieval one. The four AI Hub domains currently drop none: the upstream
preprocessor already writes only QA rows whose document reached `documents.jsonl`.

Example ids are a digest of document and question, so reports stay comparable across
runs regardless of row order. `--max-examples` with `--seed` samples deterministically;
the full four-domain set is about 236,000 questions, which is far more than a routine
run needs. `--with-filters` attaches the QA row's `legal_domain` and `document_type` as
retrieval filters, which measures retrieval within a known category rather than the
harder unfiltered task.

Recall@k here is document-level: a hit counts when any retrieved chunk belongs to the
question's source document.

## Index compatibility

BM25 splits its state across index time and query time. `document_vector` encodes stored
vectors with the indexing tokenizer and the corpus `avg_len`; `query_vector` tokenizes the
query now and looks each term up in the document-frequency map, keeping only terms it
finds. Analyze a query differently than the index was analyzed and nearly every term
misses, so the sparse leg returns nothing while dense retrieval keeps returning plausible
hits and hides the failure.

`build_hybrid_retriever` therefore validates the current settings against `index.json`
before returning a retriever, and fails closed:

```
IndexCompatibilityError: index at data/indexes/ip-v1 does not match the current settings:
  tokenizer: index built with 'regex', settings say 'kiwi'
Restore the recorded settings, or rebuild with scripts/build_index.py.
```

Every mismatch is reported at once. The embedder is checked too, though it is already
partly self-protecting: it appears in the collection name, so a different embedder reads a
different collection rather than corrupting one.

Note that `sparse.k1` and `sparse.b` take effect only at index time. `BM25Stats.load` reads
them back from `bm25.json`, so editing them in configuration changes nothing until the
index is rebuilt.

## Multi-corpus fan-out

Each corpus is its own collection with its own BM25 statistics, because BM25 is
corpus-global. To answer one question from several corpora, name them:

```yaml
retrieval:
  corpora: [criminal-v1, civil-v1, administrative-v1, ip-v1]
  corpus_weights: {ip-v1: 0.5}   # absent corpora weigh 1.0
  k_fused: 50                    # raise this when fanning out
```

`build_hybrid_retriever` then builds a dense and a sparse leg per corpus, named
`<corpus>:dense` and `<corpus>:sparse`, and fuses all of them in one weighted RRF pass.
`HybridRetriever` already accepted an arbitrary list of legs, so nothing else changed —
the orchestrator, the API, `search.py`, and `eval_retrieval.py` are untouched. An empty
`corpora` keeps single-corpus behaviour.

Each corpus's sparse leg uses only its own `bm25.json`. A term absent from one corpus's
document frequencies yields nothing from that corpus while the corpus that knows the term
still answers — which is correct, and is precisely what a single shared collection would
get wrong.

**Weights exist because rank fusion has no notion of relevance.** Every list contributes
`1/(k + rank)`, so each corpus's top hit outranks the relevant corpus's second hit no
matter how poor it is. Fanning out therefore trades precision for recall. Weights mitigate
it; the cross-encoder reranker (`rerank.backend: cross_encoder`) is the real fix. Measure Recall@k with
`corpora: []` and again with all four before concluding anything about fusion quality.

`k_fused` defaults to 20, which is tight when eight lists feed the fuser rather than two.

Diagnostics: `eval_retrieval.py --leg sparse` now means every sparse leg across every
corpus, fused, and the report header records which leg it measured. Use `close_legs(retriever)`
to release stores — fan-out opens one per corpus, and in embedded mode each holds an
exclusive lock on its index directory, which is why server mode is preferred for fan-out.

## Reranker precision and device

Measured 2026-09-27 on the RTX 4060 (8 GB): `BAAI/bge-reranker-v2-m3` scoring 50 civil
judgment chunks (2,855 characters on average) at `max_length` 512 and batch 16.

| Device / dtype | Seconds per query | Peak VRAM | Top 10 vs float32 |
|---|---|---|---|
| cuda / float32 | 2.9–3.1 | 2.5 GB | — |
| cuda / float16 | 0.8 | 1.2 GB | identical |
| cpu / float32 (16 cores) | 35–37 | 0 | identical |

`local.yaml` runs float16 on cuda. The card also holds the chat model (qwen3:4b, ~3.4 GB)
and bge-m3 (~1.9 GB), so float32 does not fit alongside them. CPU is too slow for chat,
where the agent may search more than once per turn.


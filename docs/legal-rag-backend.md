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

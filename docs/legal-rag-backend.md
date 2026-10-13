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
| `GET /v1/document` | `?chunk_id=` | the whole document that holds the chunk (`text`), and where the chunk lies in it (`cited_start`, `cited_end`); 404 if no searched corpus holds it |
| `POST /v1/cases` | `{"query", "k"}` | the best passage of each case (`hits`, `labels`) and `statutes`: per document, the articles its whole text cites, limited to laws in `retrieval.law_corpus` |
| `GET /v1/law` | `?name=` | the law's articles in order: `article`, `title`, `text`, `chunk_id`; 404 for an unknown name |
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
| `llm.num_ctx` | unset (Ollama's 4096) | `local.yaml` sets 12288: answer prompts measure up to 6,600 tokens, and 9,250 with the user's case documents, plus up to 2,048 output tokens (2026-10-06). `LEGAL_RAG__LLM__NUM_CTX` overrides it for both apps |
| `rag.allow_clarification` | `true` | Ask for facts instead of answering, only when the analyzer finds no legal area to search. Otherwise the turn is answered and the open questions ride in the answer (2026-10-05) |
| `rag.llm_rewrite` | `false` | Add LLM-generated search phrasings; no measurable retrieval gain (below) |
| `rag.analyzer_terms_in_search` | `false` | Add the analyzer's issues and keywords as a search query. Off since 2026-10-05: they made the same question retrieve different chunks from run to run (identical evidence on 55 of 80 formal and 29 of 60 everyday questions, against 79 of 80 and 60 of 60 without) for no change in gold coverage |
| `rag.hard_filter` | `false` | Use the analyzer's single needed document kind as a hard filter; harmful (below) |
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

## Gemini chat backend

The API (`scripts/serve.py`), interactive chat (`scripts/chat.py`), and evaluation
commands using `settings_from_options` read these chat settings from the project
`.env` (or the `--root` directory):

```dotenv
LEGAL_RAG__LLM__BACKEND=gemini
LEGAL_RAG__LLM__MODEL=gemini-3.7-flash
GEMINI_API_KEY=
LEGAL_RAG__LLM__GEMINI_THINKING_LEVEL=low
LEGAL_RAG__LLM__MAX_TOKENS=8192
```

Fill in the API key from Google AI Studio, then start the API with
`uv run python scripts/serve.py --profile local`. Gemini uses the existing
structured answer contract, citation verification, and local retrieval pipeline.
Questions and retrieved passages are sent to Google's GenerateContent API.
The Flask chat model still uses its separate `LLM_PROVIDER` setting.

Gemini's token cap covers reasoning and answer output together. The 2026-09-30
smoke run cut a JSON answer at 2,048 tokens, so the Gemini trial uses low reasoning
and an 8,192-token total cap. The model ID remains editable; 3.8 Flash returned
repeated HTTP 503 responses during that trial, while 3.7 Flash answered successfully.

To select the local generator, set the backend to `ollama`, model to
`qwen3:4b`, and max tokens to `2048`. Explicit CLI chat settings override the `.env` entries; exported
`LEGAL_RAG__LLM__...` environment variables have highest priority. API keys are
excluded from configuration hashes and are rejected in YAML profiles.

The 2026-09-30 frozen-evidence trial generated all 80 Gemini answers without
recorded API failures. Reference agreement was 59–60/80 (73.75–75.00%), compared
with the historical local model's 50/80 (62.5%). One local judge response remained
unresolved after retries, so the Gemini result is a range. This is an exploratory
development comparison; the paired uncertainty can include zero improvement.
Artifacts and limits are recorded in
`data/eval/reports/chat_pipeline/gemini-2026-09-30/README.md` (local, ignored by Git).

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
  corpora: [criminal-v1, civil-v1, administrative-v1, ip-v1, statute-api-full-v1]   # local.yaml since 2026-10-06
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

## Chat pipeline evaluation, 2026-09-27

Components of `/v1/chat`, run under the local profile with qwen3:4b at `num_ctx` 8192.

**Two bugs were fixed before measuring.**
- The analyzer names the native `court_decision` type. As an exact `document_type` filter that
  matched no imported corpus, so every such turn answered with zero evidence. The filter now
  uses the citation kind.
- `build_chat_model` passed no `num_ctx`, so answer prompts (5,900–6,300 tokens) were cut at
  Ollama's default 4096 without a warning.

**Analysis and rewriting against retrieval.** 210 questions: 50 each from the administrative,
civil and criminal pilot-200 sets, plus crossdomain-60. ip is left out because it sits at the
exact-gold floor. Scoring is gold document in the top 10; intervals are paired bootstraps.

| Condition | S@10 | MRR@10 | ΔS@10 95% CI vs question only |
|---|---|---|---|
| Question only (`/v1/search`) | 0.557 | 0.399 | — |
| + rule rewrite | 0.567 | 0.397 | [−0.010, +0.029] |
| + LLM rewrite | 0.571 | 0.396 | [−0.005, +0.038] |
| + rule rewrite + analyzer hard filter | 0.414 | 0.309 | [−0.205, −0.086] |

- **The hard filter:** it fired on 207 of 210 questions, and 201 of those were statute-only.
  That is why `rag.hard_filter` is now off.
- **Clarification:** the analyzer asked for more facts on 2 of 210 questions.
- **Scope:** these are formal, document-derived questions. Conversational phrasing, which the
  rewriter targets, is untested.

**Verification policy.** 80 questions: 20 each from administrative, civil, criminal and
crossdomain.
- **First drafts:** the verifier flagged 10 (12.5%). There were 10 `source_id_mismatch` issues
  (the right authority under the wrong source id) and 6 `statute_not_in_evidence`.
- **Policy outcome:** all 10 were regenerated. 4 then passed, 3 were stripped and answered, and
  3 were rejected, a 3.75% refusal rate.
- **Independent check:** `citation_guard` on the answer text flagged 7 of 80 first drafts and
  4 of 77 final answers.
- **Hand check of the 5 remaining flags:** 2 were genuinely unsupported. Both were in
  `legal_issues` or `additional_information_needed`, which the verifier and strip then skipped
  (fixed since). The other 3 were checker artefacts: a shortened law name, and the generic word
  "법령" read as a law name (fixed since).
- **Latency:** 24.8 s per turn on average.

### Follow-up, 2026-09-28

Both runs used `scripts/eval_chat_pipeline.py`. Raw results are under `data/eval/reports/chat_pipeline/`.

**Everyday phrasing.** `data/eval/everyday-150.jsonl` rewrites the same 150 administrative,
civil and criminal questions the way a person would ask them. Each keeps its gold document,
and 30 were reviewed by hand.

| Comparison | S@10 | 95% CI |
|---|---|---|
| Formal → everyday wording, question only | 0.553 → 0.480 | [−0.127, −0.020] |
| Everyday: question only → rule rewrite | 0.480 → 0.473 | [−0.033, +0.013] |
| Everyday: question only → LLM rewrite | 0.480 → 0.480 | [−0.027, +0.027] |
| Everyday: question only → + analyzer hard filter | 0.480 → 0.393 | [−0.153, −0.020] |

- **Rewriting:** everyday wording costs about 7 points, and LLM rewriting recovers none of it,
  so `llm_rewrite` stays off.
- **Clarification:** the analyzer asks for more facts on 24 of 150 everyday questions (16%),
  against 1% for formal ones.

**Verification rerun** on the same 80 questions, after the verifier and strip began covering
every list field. Generation is sampled, so single-digit differences from the first run are
within noise.
- **Outcomes:** 78 answered, 0 rejected (3 in the first run), and 2 `GenerationError`. In the
  errors, one field of the answer JSON ran past `max_tokens` 2048 and the JSON was cut, so the
  exception left `handle()` and `/v1/chat` would return 500.
- **Verifier:** it flagged 9 of 78 first drafts.
- **citation_guard:** it flagged 3 of 78 first drafts and 2 of 78 final answers.
  - In one, a stripped statute still appeared in the model-written `uncertainty`, which strip
    does not clean.
  - In the other, the verifier accepted "지방자치법 제124조" and the guard did not. The saved
    run keeps no evidence text, so it is unresolved which checker is right.


**Fixes after the rerun**
- **Truncated JSON:** when the answer JSON is cut off, the repair retry now asks for a
  shorter answer instead of a schema fix. If generation still fails, `handle()` strips
  the earlier draft when there is one and otherwise returns `kind="rejected"` with
  `reason="answer generation failed"`. It no longer raises.
- **Strip:** strip now also removes tainted sentences from the model-written `uncertainty`.
  The strip note still names the removed citations, so the eval script leaves the note
  out of the guard check.
- **지방자치법 제124조:** the verifier was right. The evidence says "「지방자치법」 제124조의
  규정을 준용한다", and the guard's `(?!의)` lookahead treated the particle 의 like the
  branch number in 제3조의3. It is now `(?!의\d)`. Verify records now keep the evidence text.
- **Recheck:** the 4 affected questions were rerun live. All four answered, and the guard
  flagged none. The two that errored before did not truncate this time, so only the unit
  tests exercise the error path.
- **Statute windows now name their article.** The imported corpora cut each statute into
  ~3,500-character windows with `metadata.article` empty. In that rerun, the chunk that
  opens in the body of 지방자치법 제124조 carried no article, so a correct `S1` citation was
  counted as `source_id_mismatch` and stripped. `Corpus.chunks()` now finds each window's
  start in the document text and names the last article heading before it
  (`legal_rag.ingestion.statute_articles`). This is not the ip-v2 structural rechunk; the
  windows and embeddings are unchanged.
  - **Coverage:** 21,229 of 21,234 statute windows across the four corpora. The other 5
    are 1940s–60s decrees whose headings have no parenthesis.
  - **Sanity check:** no window's article is numbered after the first heading inside it.
  - **Live index:** `scripts/backfill_statute_articles.py --apply` wrote the values into
    the live collections with `set_payload`, taking 53 s. Retrieval never reads the field,
    so rankings are unchanged; only the evidence header and the verifier see it.
  - **Result:** the 지방자치법 question, run 3 times, was accepted on the first draft each
    time, with no guard flags.
- **80-question rerun after the backfill.** Of 80 turns, 80 were answered, with no refusals
  and no errors. The Verifier flagged 10 first drafts, but only one was a statute mismatch
  (against 5 in the previous run). The other issues were references to non-statute
  sources, such as a date or "국가인권위원회 결정", that no source text can match. The
  guard flagged 0 first drafts and 0 final answers. Rerunning question-only retrieval on
  the 210 questions from 2026-09-27 returned the same top 10 for all of them, and S@10 was
  0.557 both times.

### Consistent answer grading, 2026-09-30

`scripts/judge_answers.py judge` now defaults to rubric **v2**. It grades three
dimensions independently for every generator:

- **Conclusion:** full/partial/no coverage and contradiction. A missing or different
  citation alone does not lower conclusion coverage; a changed substantive rule can.
- **Legal basis:** matching, different, missing, or not applicable to the reference.
  An identical conclusion supported by the wrong named law is a basis mismatch.
- **Evidence support:** supported, partial, unsupported, or evidence not saved.
  This uses the evidence supplied to the generator, not the reference's gold document.

The report exposes `conclusion_correct`, `reference_agreement` (conclusion plus
legal basis), and `grounded_correct` (both plus evidence support) separately. None
establish legal correctness. Missing evidence produces an unknown grounded metric.
Judge errors, stale grades, and generation errors remain unresolved; completed
no-answer outcomes count as failed answer outcomes. All-case score bounds show
how unresolved outcomes affect the denominator.

V2 requires reference and answer quotes to occur in their respective inputs and
source quotes to occur in the specified evidence passage. Text presence checks do
not prove entailment. Saved verification evidence takes priority over pair evidence;
typed `contexts.jsonl` bundles are a fallback. SFT context files are not substituted.

Use the same model, rubric, token budget, and thinking settings for both arms:

```bash
.venv/bin/python scripts/judge_answers.py judge --run data/eval/reports/chat_pipeline/formal-2026-09-28-articles --rubric v2 --no-think
.venv/bin/python scripts/judge_answers.py judge --run data/eval/reports/chat_pipeline/gemini-2026-09-30 --rubric v2 --no-think
.venv/bin/python scripts/judge_answers.py report --run data/eval/reports/chat_pipeline/gemini-2026-09-30 --rubric v2
```

V2 defaults to thinking disabled, context 12,288 and 4,096 output tokens. Thinking
can be enabled explicitly, but its decoding settings must match across arms.
No existing grades are overwritten: new files are `answer_judge-v2.json` and
`answer_judgments-v2.jsonl`. The immutable manifest includes the schema, prompt,
quote-rule version, model digest and decoding settings. Each grade fingerprints the
question, reference, answer, outcome kind and evidence. Failed cases retry up to
twice per invocation and remain retryable on later invocations. Configuration
changes require a separate run directory; changed inputs invalidate old grades.

Retry policy `quote-feedback-v1` keeps the initial rubric unchanged. On a schema or
quote-validation failure, its second attempt receives the rejected draft and all
validation errors, including field names, rejected quotes and exact matching source
indexes when a quote belongs to a different passage. Quote checks remain strict.
New attempts save raw output, parsed grades, validation failures, repair feedback
and available generation metadata in `answer_judge_attempts-v2.jsonl`, flushed after
each attempt. Earlier discarded outputs cannot be recovered.

The retry instruction/version is recorded separately in
`answer_judge_retry-quote-feedback-v1.json`; new grade records include its hash and
an invocation ID. Existing valid grades keep their original provenance and failed
cases retry. Reports show the accepted grades' retry-policy counts, so runs spanning
the repair are visible. An already-running process uses the old code: stop it before
rerunning the same commands to resume with the repair.

The successor policy `quote-selection-v2` addresses repeated paraphrasing that
survived feedback-only retries. Known schema/quote failures resume directly with
passage selection; a new failure switches its next attempt to that path. Reference,
answer and source text are split into unchanged contiguous passages with IDs. The
judge chooses schema-constrained IDs and the code copies their original text into
the same v2 grade fields before applying all existing quote checks. `A_NONE` denotes
no relevant answer passage and remains invalid for yes/partial coverage. Evidence
not saved and explicitly empty evidence remain distinct.

Passage-selection attempts record the full catalogue, selected IDs, output schema
hash, materialized grade and validation feedback in the attempt log. The policy is
recorded in `answer_judge_retry-quote-selection-v2.json`; successful earlier grades
remain saved. Preparing requests for observed failures and static review succeeded;
this policy has not yet been measured in a live grading run. Genuine classification
or schema failures remain unresolved and do not count as incorrect answers.

Policy `quote-selection-v3` limits evidence selection to eight unique IDs in both
its JSON schema and Python validation. Duplicate IDs are rejected rather than
silently removed. Requests time out after 180 seconds; a timeout is saved without
an immediate retry. Two consecutive timed-out cases stop the invocation with exit
code 1 after flushing results. Resume retains validated grades and remembers the
passage-selection path, including earlier attempts recorded before this policy.
The settings are recorded in `answer_judge_retry-quote-selection-v3.json`.

Ollama metadata endpoints can respond while inference is stalled. Inspect active
clients and server logs before recovering the worker and resuming. Server logs
also showed prompts exceeding the 12,288-token context limit: bounded output does
not resolve input truncation, so larger-context evaluation needs a separate run
with matching settings for both providers.

For independent manual calibration, export review inputs without AI grades or
provider metadata and fill the label fields while keeping inputs/fingerprints intact:

```bash
.venv/bin/python scripts/judge_answers.py review --run data/eval/reports/chat_pipeline/gemini-2026-09-30 --out /tmp/answer-review-v2.jsonl
.venv/bin/python scripts/judge_answers.py report --run data/eval/reports/chat_pipeline/gemini-2026-09-30 --rubric v2 --manual /tmp/answer-review-v2.jsonl
```

Use a blinded sample covering both models, including wrong-statute answers, missing
evidence, and their disagreements. Conclusion labels alone calibrate only the
conclusion metric; basis/support metrics require their own manual labels. Old
manual labels without input fingerprints cannot calibrate v2. Human calibration
and re-scoring have not been performed for v2; the Gemini trial figures above still
use the historical rubric. Archived replay scripts importing `SYSTEM`/`AnswerGrade`
also retain that legacy rubric; use the CLI commands above for v2 grading.

### Answer accuracy against the AIHub references, 2026-09-28

These historical results use the legacy rubric. Read them with
`scripts/judge_answers.py report --rubric legacy --run <historical-run>`.

`scripts/judge_answers.py` pairs the 80 saved answers from the article-backfill rerun with
the reference answers in the upstream `qa.jsonl`. The eval sets leave those out, so this is
the first check of answer content. The script also records whether the question's gold
document was among the turn's evidence, since each reference is written from that one
document. qwen3:8b (think on, `num_ctx` 12288) judged two things:
- whether the answer states the reference's key conclusion (yes, partial or no);
- whether anything in the answer contradicts the reference.

An answer counts as correct when it covers the conclusion fully and contradicts nothing.
This measures agreement with one reference answer, not legal correctness.

| Group | Correct | Covers at least partly | Contradicts |
|---|---|---|---|
| All 80 | 0.62 [0.52, 0.72] | 0.71 | 0.15 [0.09, 0.24] |
| Gold document in evidence (47) | 0.70 [0.56, 0.81] | 0.74 | 0.13 |
| Gold document not in evidence (33) | 0.52 [0.35, 0.67] | 0.67 | 0.18 |
| administrative / civil / criminal / crossdomain (20 each) | 0.60 / 0.45 / 0.80 / 0.65 | | 0.25 / 0.15 / 0.05 / 0.15 |

Intervals are Wilson 95%.

- **Calibration:** 31 answers, stratified by dataset and by gold-in-evidence, were graded
  by hand before any judge output was read.
  - Agreement: 26 of 31 on correct (Gwet AC1 0.698) and 28 of 31 on contradicts (AC1
    0.879). Both pass the project's AC1 ≥ 0.6 bar.
  - The judge is lenient on correct: it gave 21 of 31, against 18 by hand. Three of its
    "yes" grades were partial answers, and one claimed the answer covered a point it did
    not.
  - It is strict on contradicts: it gave 5 of 31, against 2 by hand. It counts an answer
    built on a different case as a contradiction.
  - The hand-graded rates on the 31 were 0.58 correct and 0.06 contradicting.
- **Main loss:** retrieval. With the gold document in evidence, 70% of answers match the
  reference; without it, 52%.
- **Civil:** civil is lowest at 0.45. Its questions are often generic ("손해액 산정에 있어
  인정되는 기준은?") while the reference states what one court held.
- **Truncated answers:** 2 of the 80 saved answers were cut at 1,500 characters.

### More evidence per answer (k_final 5 → 8), 2026-09-28: not adopted

**Gold coverage.** `eval_chat_pipeline.py coverage` replays the stored rule-rewrite queries
through `retrieve_many`, as `/v1/chat` does, and checks whether the gold document lands in
the evidence. No generation is involved.

| Evidence | Formal, 210 questions | Everyday, 150 questions |
|---|---|---|
| 5 chunks (current) | 0.471 | 0.440 |
| 5 distinct documents | 0.486 | 0.440 |
| 8 chunks | 0.543 (+0.072, CI [+0.038, +0.110]) | 0.453 (CI [+0.000, +0.033]) |
| 10 chunks | 0.567 | 0.473 |

**Context budget.** The answer prompt was measured with Ollama's `prompt_eval_count` on 14
questions.
- **5 × 1,500 characters:** median 5.7k tokens, max 6.2k. With `max_tokens` 2048 that
  already exceeds `num_ctx` 8192 at the tail.
- **8 × 1,500 characters:** max 9.5k tokens, so it needs `num_ctx` 12288. That fits:
  qwen3:4b loaded all 37 layers beside the reranker and bge-m3, at 7.0 GB.
- **10 × 1,500 characters:** needs about 14k tokens. At `num_ctx` 20480 only 26 of 37
  layers fit on the GPU.

**Answers.** The same 80 questions were run with `k_final` 8 and `num_ctx` 12288, and
qwen3:8b judged them as in the section above.
- **Accuracy:** correct answers went from 49 to 46 of the 78 questions graded in both runs
  (paired 95% CI [−0.141, +0.051]).
- **Why no gain:** in this 80-question sample, the gold document entered the evidence for
  48 questions against 47. The larger coverage gain on the 210-question set did not show
  up here.
- **Other effects:** regenerations fell from 10 to 4, turns took 37.4 s instead of 32.7 s,
  and there were no errors or truncations.
- **Judge failures:** the judge returned empty output for 2 answers, when its thinking
  used up `max_tokens`.

**Decision.** `k_final` stays at 5.

## Smaller structural civil chunks

The prepared `civil-struct800-v1` corpus uses 800/1000-character structural
chunks. The optional `structural-civil` profile expands adjacent civil passages
after reranking under a rendered-context budget, with separate citations.
See [indexing and comparison instructions](structural-civil.md). Quality gains
remain unmeasured; the existing civil-v1 index remains the baseline.

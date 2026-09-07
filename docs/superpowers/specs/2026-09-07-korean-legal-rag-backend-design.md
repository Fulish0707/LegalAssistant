# Korean Legal RAG Backend — Structural Design

Date: 2026-09-07
Status: Approved for implementation of V0.1–V0.3
Source brief: `legal_rag_assistant_project_overview.md` (user-provided, 2026-09-07)

This document is the structural design the brief asked for before any implementation. It answers the brief's 17 required outputs (section 34) and its 20 structural questions (section 33). It is written for handoff to a coding agent; the companion implementation plan is `docs/superpowers/plans/2026-09-07-korean-legal-rag-backend.md`.

## 1. System architecture

### 1.1 Decision: plain Python stages behind Protocols

The backend is a set of small, typed stages. Each stage is a class that satisfies a `typing.Protocol`, takes Pydantic models in, and returns Pydantic models out. One `Orchestrator` composes them for a chat turn. There is no workflow framework in the core.

Rejected alternatives:

- LangGraph from day one. Already on `main` for the older app, but it couples every stage to graph state and makes retrieval and verification harder to test alone. It can wrap the orchestrator later if branching control flow grows.
- Haystack or LlamaIndex. Fast to demo, but their chunkers and retrievers conflict with legal-aware chunking and deterministic verification, which the brief makes mandatory.

### 1.2 Relationship to the existing repository

The new backend lives in `src/legal_rag/` as an independent package with its own `pyproject.toml`. The existing Flask application (`app.py`, `langchain_config.py`, `static/lc-*.js`) is untouched. Once the new API is proven, the `lc-*.js` shell is pointed at it and the Flask modules are retired. Nothing in `legal_rag` imports from the Flask app.

### 1.3 Subsystem map

```text
                 ┌────────────────────────────────────────────────────────┐
                 │  api/  (FastAPI)          scripts/  (CLI entry points) │
                 └───────────────┬────────────────────────┬───────────────┘
                                 │                        │
                         ┌───────▼────────┐      ┌────────▼─────────┐
                         │ rag/           │      │ ingestion/       │
                         │ Orchestrator   │      │ chunking/        │
                         │ analyzer       │      │ (offline build)  │
                         │ rewriter       │      └────────┬─────────┘
                         │ context        │               │ writes
                         │ generator      │      ┌────────▼─────────┐
                         │ conversation   │      │ data/processed/  │
                         └──┬─────┬─────┬─┘      │ + Qdrant index   │
                            │     │     │        └──────────────────┘
               ┌────────────▼┐ ┌──▼───┐ ┌▼─────────────┐
               │ retrieval/  │ │ llm/ │ │ verification/│
               │ dense/sparse│ │      │ │              │
               │ fusion      │ └──────┘ └──────────────┘
               │ reranking/  │
               └──────┬──────┘
                      │
               ┌──────▼──────┐        ┌──────────────┐
               │ embeddings/ │        │ evaluation/  │  (reads retrieval/, rag/)
               └─────────────┘        └──────────────┘

               schemas/ and config/ are imported by everything; they import nothing above.
```

Dependency direction is strictly downward: `api` → `rag` → (`retrieval`, `llm`, `verification`) → `embeddings` → `schemas`/`config`. `ingestion` and `chunking` depend only on `schemas`/`config`. `evaluation` depends on `retrieval` and `rag` but nothing depends on it. `training` (V0.4) will depend on `schemas` and `llm` only.

## 2. Repository tree

```text
pyproject.toml                  # package legal_rag, Python 3.11, uv-managed
configs/
  default.yaml                  # every setting with a safe local default
  local.yaml                    # overrides for the Ubuntu + RTX 4060 host
  colab.yaml                    # overrides for Colab (paths, batch sizes)
data/                           # git-ignored except .gitkeep and fixtures
  raw/                          # immutable AI Hub downloads, never written by code
  processed/<corpus_version>/   # documents.jsonl, chunks.jsonl, manifest.json
  indexes/<corpus_version>/     # embedded Qdrant storage + sparse idf.json
  eval/                         # retrieval.jsonl, generation.jsonl
scripts/
  ingest.py                     # raw → processed corpus
  build_index.py                # processed → Qdrant + sparse stats
  search.py                     # query → top-k chunks (debug)
  chat.py                       # CLI chat, the V0.3 milestone interface
  eval_retrieval.py             # retrieval metrics over data/eval/retrieval.jsonl
src/legal_rag/
  __init__.py
  config/        settings.py, loader.py
  schemas/       documents.py, chunks.py, query.py, evidence.py, answer.py,
                 verification.py, evaluation.py, conversation.py
  ingestion/     loaders.py, detect.py, normalize/ (aihub_judgment.py,
                 statute_json.py, admin_interpretation.py), validate.py,
                 dedupe.py, pipeline.py, manifest.py
  chunking/      base.py, statute.py, decision.py, fallback.py, router.py
  embeddings/    base.py, ollama.py, fake.py
  retrieval/     tokenizer.py, sparse.py, store.py, dense.py, fusion.py,
                 filters.py, hybrid.py
  reranking/     base.py, noop.py, cross_encoder.py
  llm/           base.py, ollama.py, fake.py, prompts/
  rag/           analyzer.py, clarify.py, rewriter.py, context.py,
                 generator.py, orchestrator.py, conversation.py, trace.py
  verification/  citations.py, verifier.py, policy.py
  evaluation/    datasets.py, retrieval_metrics.py, retrieval_runner.py
  api/           app.py, routes.py, models.py, deps.py
  training/      (V0.4) adapters.py, sft_dataset.py, configs
tests/legal_rag/
  fixtures/      aihub_judgment_sample.json, statute_sample.json,
                 admin_interpretation_sample.json, eval_retrieval_sample.jsonl
  unit/ ...      one file per module
  integration/   test_ingest_to_search.py, test_chat_turn.py
  live/          test_ollama_live.py (skipped unless LEGAL_RAG_LIVE=1)
```

The existing top-level `tests/` directory belongs to the Flask app. New tests live under `tests/legal_rag/` and are selected by `pytest tests/legal_rag`.

## 3. Module responsibilities

| Module | Owns | Must not do |
| --- | --- | --- |
| `config` | Load YAML profile, apply env overrides, expose one immutable `Settings` object | Read secrets from YAML |
| `schemas` | All Pydantic contracts shared across modules | Import any other `legal_rag` module |
| `ingestion` | Raw file discovery, schema detection, normalization to canonical documents, validation, dedup, manifest | Chunk, embed, or touch Qdrant |
| `chunking` | Turn one canonical document into ordered `Chunk`s with structural paths | Compute embeddings |
| `embeddings` | Map text to dense vectors behind `Embedder` | Know about Qdrant |
| `retrieval` | Tokenization, BM25 statistics, Qdrant store, dense and sparse retrievers, RRF fusion, filter translation | Call the LLM |
| `reranking` | Reorder `ScoredChunk`s behind `Reranker` | Retrieve |
| `llm` | `ChatModel` Protocol and backends; prompt templates as files | Build evidence or verify |
| `rag` | Query analysis, clarification policy, rewriting, evidence building, generation, orchestration, conversation store, trace | Parse raw data |
| `verification` | Deterministic citation parsing and grounding checks, handling policy | Generate text |
| `evaluation` | Dataset schemas, metrics, runners | Mutate indexes |
| `api` | FastAPI app, request/response models, dependency wiring | Business logic |
| `training` (V0.4) | SFT dataset generation, QLoRA config, adapter registry | Serve requests |

## 4. Core interfaces

All Protocols live next to their first implementation and are re-exported from the module `__init__`.

```python
# embeddings/base.py
class Embedder(Protocol):
    name: str            # e.g. "ollama:bge-m3"; stamped into collection names
    dimension: int
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...

# retrieval/tokenizer.py
class Tokenizer(Protocol):
    name: str
    def tokenize(self, text: str) -> list[str]: ...

# retrieval — shared result type (schemas/chunks.py)
class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    retriever: str       # "dense" | "sparse" | "fused" | "reranked"

# retrieval/dense.py, retrieval/sparse.py, retrieval/hybrid.py
class Retriever(Protocol):
    name: str
    def retrieve(self, query: str, k: int, filters: RetrievalFilters | None = None) -> list[ScoredChunk]: ...

# retrieval/fusion.py
class Fuser(Protocol):
    def fuse(self, ranked_lists: list[list[ScoredChunk]], k: int) -> list[ScoredChunk]: ...

# reranking/base.py
class Reranker(Protocol):
    name: str
    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]: ...

# llm/base.py
class ChatModel(Protocol):
    name: str
    def generate(self, messages: list[Message], *, response_schema: type[BaseModel] | None = None,
                 temperature: float = 0.2, max_tokens: int = 1024) -> ModelResponse: ...

# rag/conversation.py
class ConversationStore(Protocol):
    def get(self, conversation_id: str) -> Conversation: ...
    def append(self, conversation_id: str, turn: Turn) -> None: ...
```

Stage classes in `rag/` are concrete but each depends only on the Protocols above, so any backend can be swapped through configuration.

### Answers to structural questions 4–7

- **Retrieval interfaces (Q4):** `Retriever`, `Fuser`, `Reranker`, plus `VectorStore` inside `retrieval/store.py` wrapping `qdrant_client`. Only `store.py` imports Qdrant.
- **Where filtering occurs (Q5):** `RetrievalFilters` is a schema object. `retrieval/filters.py` translates it into a Qdrant `Filter` applied at query time in both dense and sparse searches. Post-fusion filtering is never needed because both legs are pre-filtered.
- **Fusion (Q6):** Reciprocal rank fusion, `score = Σ 1 / (rrf_k + rank)`, `rrf_k = 60` by default, over the dense and sparse lists, deduplicated by `chunk_id`. Configurable weights per leg.
- **Reranker return (Q7):** the same `list[ScoredChunk]`, truncated to `k`, with `score` replaced by the reranker's score and `retriever="reranked"`. The pre-rerank score is kept in `chunk.metadata["fused_score"]` for tracing.

## 5. Data schemas

All schemas are Pydantic v2 models with `model_config = ConfigDict(frozen=True, extra="forbid")` unless noted. Field names are English; text content is Korean.

### 5.1 Documents (`schemas/documents.py`)

```python
class DocumentType(StrEnum):
    statute = "statute"
    court_decision = "court_decision"
    admin_interpretation = "admin_interpretation"

class SourceRef(BaseModel):
    dataset: str          # "aihub_judgment_v1"
    file: str             # relative path under data/raw
    record_key: str       # natural key inside the file, e.g. case number
    url: str | None = None

class LegalDocumentBase(BaseModel):
    id: str               # "{document_type}:{sha256(dataset + record_key)[:16]}"
    document_type: DocumentType
    title: str
    text: str             # full canonical text, used for fallback chunking and display
    jurisdiction: str = "KR"
    legal_domain: str | None = None      # "민사", "형사", "행정" ...
    source: SourceRef
    created_date: date | None = None
    effective_date: date | None = None
    metadata: dict[str, str | int | float | bool | None] = {}

class StatuteItem(BaseModel):    # 호
    number: str; text: str
class StatuteParagraph(BaseModel):  # 항
    number: str; text: str; items: list[StatuteItem] = []
class StatuteArticle(BaseModel):    # 조
    number: str            # "제750조", "제3조의2"
    title: str | None
    text: str              # article body excluding paragraphs, may be empty
    paragraphs: list[StatuteParagraph] = []
    chapter: str | None = None

class Statute(LegalDocumentBase):
    document_type: Literal[DocumentType.statute]
    law_name: str
    version: str | None          # law.go.kr 법령ID/공포번호 when available
    promulgation_date: date | None
    articles: list[StatuteArticle]

class DecisionSection(StrEnum):
    issues = "issues"; facts = "facts"; reasoning = "reasoning"; holding = "holding"; summary = "summary"

class CourtDecision(LegalDocumentBase):
    document_type: Literal[DocumentType.court_decision]
    court: str; case_number: str; decision_date: date | None; case_type: str | None
    sections: dict[DecisionSection, str]     # only sections present in the source
    referenced_statutes: list[str] = []      # "민법 제750조"
    referenced_cases: list[str] = []         # case numbers cited by the decision

class AdminInterpretation(LegalDocumentBase):
    document_type: Literal[DocumentType.admin_interpretation]
    authority: str; reference_number: str; date: date | None
    subject: str; question: str; interpretation: str; legal_basis: list[str] = []

LegalDocument = Annotated[Statute | CourtDecision | AdminInterpretation, Field(discriminator="document_type")]
```

**Q1 (exact contracts):** the models above are the contract. `text` on the base is always populated by the normalizer as a deterministic rendering of the structured fields so that every document type has one flat text for fallback chunking and display.

**Q18 (temporal versions):** a statute version is a distinct `Statute` document with the same `law_name` and a different `version` and `effective_date`. The document id includes the version through `record_key`. `RetrievalFilters.effective_on` selects the version whose `effective_date <= date` and whose `metadata["repealed_date"]` is null or later. V0.1 stores the fields; V0.5 implements the date selection.

### 5.2 Chunks (`schemas/chunks.py`)

```python
class Chunk(BaseModel):
    id: str                   # "{document_id}#{path}"
    document_id: str
    document_type: DocumentType
    path: str                 # "art:제750조", "art:제3조의2/para:1", "sec:reasoning/2"
    order: int                # position within the document
    text: str                 # retrieval text; includes a short structural header line
    title: str                # document title for display
    metadata: ChunkMetadata

class ChunkMetadata(BaseModel):
    law_name: str | None; article: str | None; paragraph: str | None
    court: str | None; case_number: str | None; decision_date: date | None
    authority: str | None; reference_number: str | None
    legal_domain: str | None; effective_date: date | None; section: str | None
    dataset: str; corpus_version: str
```

**Q3 (parent linking):** every chunk carries `document_id` and `path`. The processed corpus keeps `documents.jsonl` alongside `chunks.jsonl`, and `Corpus.document(document_id)` loads the parent on demand. Qdrant payload stores the full `ChunkMetadata` plus `document_id` so the parent is reachable from any hit without a second store.

### 5.3 Query analysis (`schemas/query.py`)

```python
class QueryAnalysis(BaseModel):
    domain: str | None; subdomain: str | None
    issues: list[str]; key_facts: list[str]; dates: list[str]
    keywords: list[str]
    statute_refs: list[str]        # extracted deterministically before the LLM pass
    case_refs: list[str]
    document_types_needed: list[DocumentType]
    needs_more_info: bool
    missing_facts: list[str]       # questions to ask when needs_more_info

class RetrievalFilters(BaseModel):
    document_types: list[DocumentType] | None = None
    legal_domain: str | None = None
    court: str | None = None
    law_name: str | None = None
    decided_after: date | None = None; decided_before: date | None = None
    effective_on: date | None = None

class RewrittenQueries(BaseModel):
    original: str
    search_queries: list[str]      # 1–3 legal-search phrasings, original always first
```

### 5.4 Evidence (`schemas/evidence.py`)

```python
class Evidence(BaseModel):
    source_id: str            # "S1", "S2" ... stable within one turn
    chunk_id: str
    document_id: str
    document_type: DocumentType
    header: dict[str, str]    # rendered key/values: Law, Article, Court, Case number, Date
    text: str
    score: float

class EvidenceBundle(BaseModel):
    items: list[Evidence]
    def render(self) -> str    # the [LEGAL SOURCE n] block format from the brief
```

**Q8 (id propagation):** `ContextBuilder` assigns `S1..Sn` in rank order and keeps the `EvidenceBundle` for the turn. The generator prompt tells the model to cite only these ids. The verifier receives the same bundle, so ids never leave the process boundary as anything but this object.

### 5.5 Structured answer (`schemas/answer.py`)

```python
class CitationKind(StrEnum): statute = "statute"; case = "case"; interpretation = "interpretation"

class Citation(BaseModel):
    source_id: str            # must be one of the bundle ids
    kind: CitationKind
    reference: str            # "민법 제750조" or a case number, as the model wrote it
    quote: str | None = None

class Confidence(StrEnum): low = "low"; medium = "medium"; high = "high"

class LegalAnswer(BaseModel):          # Q9: the LLM output contract
    summary: str
    legal_issues: list[str]
    analysis: str
    possible_actions: list[str]
    additional_information_needed: list[str]
    citations: list[Citation]
    uncertainty: str
    confidence: Confidence
```

`LegalAnswer` is what the model returns as JSON (Ollama `format` = the Pydantic JSON schema). The API response wraps it with evidence and trace; it is never rendered raw.

### 5.6 Verification (`schemas/verification.py`)

```python
class VerificationIssue(BaseModel):
    kind: Literal["unknown_source_id", "statute_not_in_evidence", "case_not_in_evidence", "source_id_mismatch"]
    detail: str
class VerificationReport(BaseModel):
    ok: bool
    issues: list[VerificationIssue]
    checked_statutes: list[str]; checked_cases: list[str]
class VerificationAction(StrEnum): accept = "accept"; regenerate = "regenerate"; strip = "strip"; reject = "reject"
```

### 5.7 Evaluation (`schemas/evaluation.py`)

**Q12:** JSONL, one example per line.

```python
class RetrievalExample(BaseModel):
    id: str; query: str
    relevant_document_ids: list[str]
    relevant_chunk_ids: list[str] = []      # optional finer grain
    filters: RetrievalFilters | None = None
class GenerationExample(BaseModel):          # used from V0.4
    id: str; query: str; history: list[str] = []
    expected_issues: list[str] = []; expected_statutes: list[str] = []; expected_cases: list[str] = []
    expect_clarification: bool = False
    forbidden_references: list[str] = []     # authorities that must not be confirmed
```

### 5.8 Conversation (`schemas/conversation.py`)

```python
class Turn(BaseModel):
    role: Literal["user", "assistant"]; content: str; created_at: datetime
    evidence_ids: list[str] = []      # chunk ids used, for audit only
class Conversation(BaseModel):
    id: str; turns: list[Turn]
```

**Q17:** history text never enters retrieval. The orchestrator passes at most the last `conversation.history_turns` (default 4) user/assistant turns to the query analyzer only, as a `history` field. Evidence is rebuilt each turn from the current retrieval and is never stored as turn content, only as `evidence_ids` for audit.

## 6. Data flow (offline)

```text
data/raw/<dataset>/**            immutable
   │ ingestion.loaders.discover()            → RawRecord(dataset, file, payload, sha256)
   │ ingestion.detect.detect_schema()        → "aihub_judgment_v1" | "statute_json_v1" | "admin_interpretation_v1"
   │ ingestion.normalize.<schema>.normalize()→ LegalDocument (or NormalizationError)
   │ ingestion.validate.validate()           → drops documents failing invariants, records reason
   │ ingestion.dedupe.dedupe()               → first-seen wins on (dataset, record_key), then on text hash
   ▼
data/processed/<corpus_version>/documents.jsonl
   │ chunking.router.chunk_document()        → list[Chunk]
   ▼
data/processed/<corpus_version>/chunks.jsonl
data/processed/<corpus_version>/manifest.json
   │ scripts/build_index.py
   │   retrieval.sparse.fit_bm25(chunks)     → idf.json + doc lengths
   │   embeddings.Embedder.embed_documents   → dense vectors
   │   retrieval.store.upsert                → Qdrant collection legal_<corpus_version>_<embedder_slug>
   ▼
data/indexes/<corpus_version>/
```

**Q2 (normalization versioning):** `manifest.json` records `corpus_version` (user-chosen label), `pipeline_version` (a constant in `ingestion/__init__.py`, bumped on any behavior change), the config hash, and for every raw file its path, size, and sha256, plus counts of loaded, normalized, dropped, and deduplicated records with reasons. Rebuilding with the same raw files and pipeline version must produce byte-identical `documents.jsonl`; a test enforces this on fixtures.

Schema detection uses key fingerprints. Fixture layouts are assumptions until real AI Hub files are available (see risk R1):

- `aihub_judgment_v1`: top-level keys `info`, `facts`, `dcss`, `close` with `info.caseNo`, `info.courtNm`, `info.judmnAdjuDe`.
- `statute_json_v1`: `law_name`, `articles[]` with `number`, `title`, `paragraphs[]`.
- `admin_interpretation_v1`: `authority`, `reference_number`, `question`, `interpretation`.

## 7. Retrieval flow (online)

```text
search_queries (from rewriter)   RetrievalFilters (from analyzer)
        │                                   │
        ├── DenseRetriever.retrieve(q, k_dense, filters)   Qdrant named vector "dense"
        └── SparseRetriever.retrieve(q, k_sparse, filters) Qdrant named sparse vector "bm25"
                       │
              RRFFuser.fuse([...all lists...], k_fused)   dedupe by chunk_id
                       │
              Reranker.rerank(original_query, fused, k_final)   NoopReranker in V0.2, CrossEncoderReranker optional
                       │
                 list[ScoredChunk]  → ContextBuilder
```

Defaults: `k_dense = k_sparse = 20`, `k_fused = 20`, `k_final = 5`. Each rewritten query runs both legs; RRF fuses all lists at once, which naturally rewards chunks hit by several phrasings.

Sparse leg details. `retrieval/tokenizer.py` ships `RegexTokenizer`, which preserves legal identifiers as single tokens (`제750조`, `제3조의2`, case numbers such as `2019다12345`, `2020도1234`) and emits Hangul character bigrams for other words so recall does not depend on morphology. `KiwiTokenizer` is an optional drop-in when `kiwipiepy` is installed. BM25 document-side weights (`k1 = 1.2`, `b = 0.75`) are stored as Qdrant sparse vectors; the query-side vector carries idf, so Qdrant's dot product is the BM25 score. `idf.json` and average document length are written at index time and loaded by `SparseRetriever`.

Collection naming: `legal_{corpus_version}_{embedder_slug}`. Switching embedder creates a new collection; the old one remains.

## 8. Generation flow

```text
user message + last N turns
   │ QueryAnalyzer
   │   1. deterministic pass: verification.citations.extract_refs(message) → statute_refs, case_refs
   │   2. LLM pass with response_schema=QueryAnalysis (history included as plain text)
   │   3. merge: deterministic refs override the model's
   ▼ QueryAnalysis
   │ ClarificationPolicy
   │   if analysis.needs_more_info and len(missing_facts) > 0 and settings.rag.allow_clarification:
   │       return ChatResponse(kind="clarification", questions=missing_facts)   # no retrieval
   ▼
   │ QueryRewriter
   │   deterministic: original + "<law_name> <article>" for each statute ref
   │   LLM pass (optional, settings.rag.llm_rewrite): up to 2 more legal-search phrasings
   ▼ RewrittenQueries
   │ HybridRetriever + Reranker  (section 7)
   ▼ list[ScoredChunk]
   │ ContextBuilder → EvidenceBundle (S1..Sn), rendered as the brief's [LEGAL SOURCE n] blocks
   ▼
   │ Generator
   │   messages = [system prompt (llm/prompts/answer_system.md), evidence block, user message]
   │   ChatModel.generate(response_schema=LegalAnswer)
   │   on JSON/schema failure: retry once with the validation error appended; then raise GenerationError
   ▼ LegalAnswer
   │ Verifier + VerificationPolicy (section 9)
   ▼
ChatResponse(kind="answer", answer=LegalAnswer, evidence=EvidenceBundle, verification=VerificationReport, trace=Trace)
```

**Q13 (model backends):** `ChatModel` has `OllamaChatModel` (HTTP `/api/chat`, `format` set to the JSON schema, `think` disabled) and `FakeChatModel` (scripted responses for tests). V0.4 adds `TransformersChatModel(base, adapter=None)` so experiments A–D are pure configuration: `llm.backend`, `llm.model`, `llm.adapter`, and `rag.enabled`.

**Q16 (sync vs async):** the core is synchronous. Ollama calls are blocking HTTP; Qdrant embedded mode is synchronous. FastAPI routes are `async def` and run the orchestrator through `run_in_threadpool`. Streaming is not in scope before V1.0.

## 9. Verification flow

**Q10:** verification runs inside the orchestrator immediately after the structured output is parsed and before anything is rendered. It never runs inside the generator, so the generator can be evaluated unverified for experiments.

Checks in V0.3 (all deterministic, `verification/verifier.py`):

1. `unknown_source_id`: a `citation.source_id` not in the bundle.
2. `source_id_mismatch`: the citation's `reference` names a statute or case that the cited `S`-block does not contain.
3. `statute_not_in_evidence`: any `<law> 제N조` pattern in `analysis` or `summary` whose article is not in any evidence header or text.
4. `case_not_in_evidence`: any case-number pattern in the answer text not present in the evidence.

Two deterministic refinements were added after the first live run with `qwen3:4b`:

- **Citation backfill** (`verification/backfill.py`): when the answer prose names an article or case number that an evidence block contains, a citation pointing at that block is added before verification. Nothing outside the bundle is ever added; this only restores traceability the model dropped.
- **User-mentioned references are exempt from the not-in-evidence checks.** If the user wrote `민법 제839조의2` in the question, the answer may repeat it to say it could not be confirmed. Citing it with a source id is still checked against that block (`source_id_mismatch`).

V0.5 adds `temporal_mismatch` (cited version not effective on the event date) and `unsupported_claim` (sentence-level entailment against the cited block).

**Q11 (failed verification):** `VerificationPolicy.decide(report, attempt)`:

```text
report.ok                         → accept
attempt == 0 and issues non-empty → regenerate (feedback message lists each issue; temperature unchanged)
attempt == 1 and issues non-empty → strip: remove offending citations, delete sentences containing
                                    unsupported references, set confidence = low, append the issues to
                                    `uncertainty`
strip leaves no citations and the query needed evidence → reject: ChatResponse(kind="rejected", reason=...)
```

Every decision is written to the trace. The policy is one class so the thresholds can change without touching the orchestrator.

## 10. Evaluation architecture

- `evaluation/datasets.py` reads and validates JSONL into `RetrievalExample` / `GenerationExample`.
- `evaluation/retrieval_metrics.py`: `recall_at_k`, `precision_at_k`, `mrr`, `ndcg_at_k`, each a pure function over a ranked id list and a relevant id set, unit-tested against hand-computed values.
- `evaluation/retrieval_runner.py`: `run(retriever, examples, ks=(5,10,20)) -> RetrievalReport` with per-example rows and aggregates, printed as a table and written to JSON. Any `Retriever` works, so dense-only, sparse-only, and hybrid are compared by config.
- Generation evaluation (V0.4) reuses the orchestrator with `FakeChatModel` off and a real model on, scoring citation accuracy, hallucinated statute and case rate, and clarification behavior from `GenerationExample` fields. The verifier's parsers are reused for scoring so scoring and enforcement cannot drift apart.

## 11. Configuration strategy

`config/settings.py` defines nested Pydantic Settings:

```text
Settings
  paths:        raw_dir, processed_dir, index_dir, eval_dir
  corpus:       version
  embedding:    backend ("ollama" | "fake"), model ("bge-m3"), base_url, batch_size
  sparse:       tokenizer ("regex" | "kiwi"), k1, b
  qdrant:       mode ("embedded" | "server"), path, url, api_key (env only)
  retrieval:    k_dense, k_sparse, k_fused, k_final, rrf_k, dense_weight, sparse_weight
  rerank:       backend ("noop" | "cross_encoder"), model, device
  llm:          backend ("ollama" | "fake" | "transformers"), model ("qwen3:4b"), base_url, temperature, max_tokens, adapter
  rag:          allow_clarification, llm_rewrite, history_turns, max_regenerations
  api:          host, port
```

Loading order: `configs/default.yaml` → `configs/<profile>.yaml` (profile from `LEGAL_RAG_PROFILE`, default `local`) → environment variables prefixed `LEGAL_RAG__` with `__` as the nesting delimiter → explicit overrides from CLI flags. Secrets (`qdrant.api_key`, future model API keys) are environment-only; the YAML loader rejects them if present. `Settings.hash()` returns a stable hash used in the manifest.

**Q15 (local vs Colab):** identical code and YAML keys; `colab.yaml` only changes paths (Drive mount), batch sizes, and the Qdrant mode. Training configs (V0.4) follow the same profile mechanism.

## 12. Environment strategy

- Python 3.11 provisioned with `uv python install 3.11`; `uv sync` creates `.venv` from `pyproject.toml` and `uv.lock`.
- Ollama on the host serves `qwen3:4b` (present) and `bge-m3` (to pull). `nomic-embed-text` remains selectable by config.
- Qdrant runs embedded (`QdrantClient(path=...)`) for development and tests. `docker compose up qdrant` is the server mode for the API once Docker socket access is available on the host; this session cannot reach the Docker socket.
- Optional extras: `rerank` (sentence-transformers, torch) and `kiwi` (kiwipiepy). Core install has no torch dependency so tests run in seconds.
- PostgreSQL is deferred to V1.0; the conversation store is in-memory with a SQLite implementation behind the same Protocol.

## 13. Development version plan

| Version | Scope in this design | Exit criterion |
| --- | --- | --- |
| V0.1 Data foundation | `config`, `schemas`, `ingestion`, `chunking`, `scripts/ingest.py`, manifest | Fixtures ingest reproducibly into `documents.jsonl` and `chunks.jsonl`; identical bytes on rerun |
| V0.2 Retrieval baseline | `embeddings`, `retrieval`, `reranking/noop`, `scripts/build_index.py`, `scripts/search.py`, `evaluation` retrieval metrics and runner, `scripts/eval_retrieval.py` | Hybrid search over the fixture index returns the expected chunk for each fixture query; metrics script prints Recall@5, MRR, nDCG@10 |
| V0.3 RAG chatbot baseline | `llm`, `rag`, `verification` (source-id, statute, case checks and policy), `scripts/chat.py`, `api` | `python scripts/chat.py` answers a fixture question with `S`-ids that verification accepts; FastAPI `/v1/chat` returns the same payload; fake-model integration test passes |
| V0.4 Legal QLoRA | `training/`, `TransformersChatModel`, SFT dataset generation, generation evaluation | Out of scope for this plan; interfaces reserved |
| V0.5 Reliability | temporal checks, claim-level grounding, richer policy | Out of scope; `VerificationIssue.kind` is extensible |

## 14. Testing strategy

- Unit tests per module under `tests/legal_rag/unit/`, using fixture files under `tests/legal_rag/fixtures/`. No network, no GPU.
- `FakeEmbedder` produces deterministic vectors from token hashes so dense retrieval is testable offline and ranking assertions are stable.
- `FakeChatModel` returns scripted `ModelResponse`s keyed by the response schema, including one deliberately hallucinated citation to exercise the verifier and policy.
- Integration tests use embedded Qdrant in a temporary directory: ingest fixtures → build index → search → chat turn.
- Live tests under `tests/legal_rag/live/` hit Ollama and are skipped unless `LEGAL_RAG_LIVE=1`.
- Reproducibility test: ingest fixtures twice into two temp dirs and compare `documents.jsonl` bytes.
- Command: `uv run pytest tests/legal_rag -q`.

## 15. Risk register

| Id | Risk | Impact | Mitigation |
| --- | --- | --- | --- |
| R1 | AI Hub file layouts differ from the fixture assumptions | Loaders fail on real data | Schema detection by fingerprint plus one normalizer per layout; adding a layout is one module and one fixture. Manifest records unrecognized files instead of failing the run |
| R2 | Korean recall of the embedding model | Poor dense hits | `bge-m3` default, sparse leg mandatory, retrieval eval from V0.2 to measure each leg |
| R3 | Qwen3 4B fails to emit valid `LegalAnswer` JSON | Generation errors | Ollama JSON-schema `format`, one repair retry with the validation error, `think` disabled |
| R4 | 8 GB VRAM shared between LLM, embedder, and reranker | OOM or slow turns | Embedder via Ollama shares the runtime; reranker optional and CPU-capable; V0.4 training unloads serving models |
| R5 | Statute article parsing edge cases (`제3조의2`, 삭제 조문, 부칙) | Wrong chunk boundaries | Regex covered by unit tests; unknown structures fall back to sentence-window chunking with `path="fallback:n"` |
| R6 | Embedded Qdrant allows one client per storage path, even inside one process | API and index build collide; tests that open a second retriever fail | Index build is a separate step and closes its client; the API builds one retriever lazily; integration tests share a single module-scoped retriever; server mode for concurrent use |
| R7 | Verification strips too aggressively and empties answers | Unhelpful responses | Reject only when no citations remain and evidence was required; trace records every strip for tuning |
| R8 | Docker socket unavailable in this session | Cannot validate server mode | Embedded mode is the tested path; server mode is covered by a live test to run on the host |

## 16. Open design decisions

| Id | Decision | Default taken now | Revisit when |
| --- | --- | --- | --- |
| D1 | Statute source layout | Internal `statute_json_v1`, law.go.kr-like | Real statute data chosen |
| D2 | Cross-encoder model | `BAAI/bge-reranker-v2-m3` when `rerank` extra installed | After V0.2 retrieval eval shows fusion-only ceiling |
| D3 | LLM query rewrite | Off by default; deterministic rewrite only | After measuring recall gain against latency |
| D4 | Conversation persistence | In-memory + SQLite | V1.0 PostgreSQL |
| D5 | Temporal law selection rule | Fields stored, no selection | V0.5 |
| D6 | Claim-level entailment check | Not implemented | V0.5, needs an NLI or judge model |
| D7 | Streaming responses | Not implemented | V1.0 |

## 17. Recommended implementation order

1. Project skeleton: `pyproject.toml`, `uv` environment, `config`, empty packages, pytest wiring.
2. `schemas` in full, with round-trip tests.
3. `ingestion`: loaders, detect, three normalizers, validate, dedupe, manifest, pipeline, `scripts/ingest.py`.
4. `chunking`: statute, decision, fallback, router.
5. `embeddings`: Protocol, fake, Ollama.
6. `retrieval`: tokenizer, BM25 fit, Qdrant store, dense, sparse, RRF, filters, hybrid; `scripts/build_index.py`, `scripts/search.py`.
7. `evaluation`: metrics, datasets, runner, `scripts/eval_retrieval.py`.
8. `llm`: Protocol, fake, Ollama, prompt files.
9. `verification`: citation parser, verifier, policy.
10. `rag`: analyzer, clarify, rewriter, context, generator, conversation, trace, orchestrator; `scripts/chat.py`.
11. `api`: FastAPI app and routes; integration tests.

Each step ends with a passing test run and a commit. The companion plan breaks these into bite-sized TDD tasks.

## Appendix: remaining structural questions

- **Q14 (adapter versioning):** V0.4 will store adapters under `models/adapters/<name>/<version>/` with an `adapter.json` recording base checkpoint, training config hash, and dataset manifest hash; `llm.adapter` selects one by `name@version`.
- **Q19 (audit logs):** `rag/trace.py` emits one JSON line per turn with `conversation_id`, timings per stage, analyzer output, rewritten queries, retrieved chunk ids with scores per leg, evidence ids, verification report, policy action, and model name. The line never contains raw user text unless `LEGAL_RAG__TRACE__INCLUDE_TEXT=1`.
- **Q20 (minimum viable tree for V0.1–V0.3):** everything in section 2 except `training/`, `reranking/cross_encoder.py`, and `evaluation` generation runners.

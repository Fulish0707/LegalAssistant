# Multi-corpus Fan-out Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Answer one query from several legal corpora at once, fusing per-corpus dense and sparse results in a single weighted RRF pass.

**Architecture:** `HybridRetriever` already queries an arbitrary list of legs, fuses them all in one pass, and reranks. Fan-out is therefore a factory change: build dense and sparse legs for each configured corpus, name each leg after its corpus so fusion can weight it, and hand the whole set to the existing class. `build_hybrid_retriever` dispatches on configuration, so the orchestrator, API, `search.py`, and `eval_retrieval.py` need no changes at all.

**Tech Stack:** Python 3.11, Pydantic v2, Qdrant (embedded for tests, server for production), pytest, uv.

**Design doc:** `docs/superpowers/specs/2026-09-15-multi-corpus-fanout-design.md`

**Run tests with `uv run pytest`.** The project is not importable from the anaconda python on this machine.

**Ordering note.** Tasks 1-6 are built and tested entirely in embedded Qdrant with small temporary corpora, so none of them is blocked on infrastructure. The Qdrant server and the production re-index come last, because re-embedding 497,000 chunks takes hours and gates nothing in the code.

**Refinement from the design doc.** The design names `build_fanout_retriever` as the entry point. Implement it, but have the existing `build_hybrid_retriever` dispatch to it when `retrieval.corpora` is non-empty. That keeps all four call sites unchanged and makes the "no caller changes" claim literally true.

---

### Task 1: Configure which corpora to query

**Files:**
- Modify: `src/legal_rag/config/settings.py` (`RetrievalSettings`)
- Modify: `configs/default.yaml` (`retrieval:` section)
- Test: `tests/legal_rag/unit/test_settings.py`

**Step 1: Write the failing test**

Append to `tests/legal_rag/unit/test_settings.py`:

```python
def test_retrieval_corpora_default_to_single_corpus_behaviour():
    settings = load_settings(profile="default")
    assert settings.retrieval.corpora == []
    assert settings.retrieval.corpus_weights == {}


def test_retrieval_corpora_and_weights_are_configurable():
    settings = load_settings(profile="default", overrides={"retrieval": {
        "corpora": ["criminal-v1", "ip-v1"], "corpus_weights": {"ip-v1": 0.5}}})
    assert settings.retrieval.corpora == ["criminal-v1", "ip-v1"]
    assert settings.retrieval.corpus_weights == {"ip-v1": 0.5}
```

Check the file's existing imports first; `load_settings` is probably already imported.

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/legal_rag/unit/test_settings.py -q`
Expected: FAIL — `ValidationError: Extra inputs are not permitted [corpora]` (the section sets `extra="forbid"`)

**Step 3: Write minimal implementation**

In `src/legal_rag/config/settings.py`, add two fields to `RetrievalSettings`, after `rrf_k`:

```python
    corpora: list[str] = Field(default_factory=list)  # empty means query corpus.version alone
    corpus_weights: dict[str, float] = Field(default_factory=dict)  # absent corpora weigh 1.0
```

In `configs/default.yaml`, document them under `retrieval:` without changing behaviour:

```yaml
  corpora: []              # e.g. [criminal-v1, civil-v1] to fan out; empty queries corpus.version
  corpus_weights: {}       # e.g. {ip-v1: 0.5}; absent corpora weigh 1.0
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/legal_rag/unit/test_settings.py -q`
Expected: PASS

Then confirm nothing else regressed, since `Settings.hash()` feeds index provenance:

Run: `uv run pytest -q`
Expected: all pass

**Step 5: Commit**

```bash
git add src/legal_rag/config/settings.py configs/default.yaml tests/legal_rag/unit/test_settings.py
git commit -m "feat: configure which corpora retrieval queries"
```

End every commit message in this plan with:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

---

### Task 2: Build per-corpus legs

**Files:**
- Modify: `src/legal_rag/factory.py`
- Test: `tests/legal_rag/unit/test_fanout.py` (create)

This task builds the legs and names them. Weighting is Task 4; dispatch is Task 5.

**Step 1: Write the test helper and the failing test**

Create `tests/legal_rag/unit/test_fanout.py`. It needs two real indexed corpora, so build them with the fake embedder in a temp root:

```python
import json

import pytest

from legal_rag.config import load_settings
from legal_rag.factory import build_corpus_legs
from legal_rag.ingestion import import_processed as importer
from legal_rag.retrieval.index_builder import build_index


def settings(tmp_path, version="a", **retrieval):
    overrides = {"paths": {"root": str(tmp_path)}, "corpus": {"version": version},
                 "embedding": {"backend": "fake"}}
    if retrieval:
        overrides["retrieval"] = retrieval
    return load_settings(profile="default", overrides=overrides)


def make_corpus(tmp_path, version, terms):
    """Index one tiny corpus whose vocabulary is `terms`, so BM25 df differs per corpus."""
    source = tmp_path / f"src-{version}"
    source.mkdir()
    docs = [dict(document_id=f"{version}-doc{i}", document_type="statute",
                 legal_domain=version, title=f"{version} 법", text=f"{term} 조문 내용",
                 metadata={}) for i, term in enumerate(terms)]
    chunks = [dict(d, chunk_id=f"{version}-chunk{i}", chunk_index=0, total_chunks=1)
              for i, d in enumerate(docs)]
    for name, rows in (("documents.jsonl", docs), ("chunks.jsonl", chunks)):
        (source / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    cfg = settings(tmp_path, version)
    importer.import_processed(source, cfg)
    build_index(cfg)
    return cfg


def test_legs_are_named_after_their_corpus(tmp_path):
    make_corpus(tmp_path, "alpha", ["특허", "상표"])
    make_corpus(tmp_path, "beta", ["절도", "사기"])
    cfg = settings(tmp_path, "alpha", corpora=["alpha", "beta"])
    legs = build_corpus_legs(cfg)
    assert [leg.name for leg in legs] == [
        "alpha:dense", "alpha:sparse", "beta:dense", "beta:sparse"]


def test_each_corpus_sparse_leg_uses_its_own_statistics(tmp_path):
    # The point of per-corpus BM25: a term only beta knows must still retrieve beta hits,
    # and must not be silently dropped because alpha's df lacks it.
    make_corpus(tmp_path, "alpha", ["특허", "상표"])
    make_corpus(tmp_path, "beta", ["절도", "사기"])
    cfg = settings(tmp_path, "alpha", corpora=["alpha", "beta"])
    legs = {leg.name: leg for leg in build_corpus_legs(cfg)}
    assert legs["beta:sparse"].retrieve("절도", 5)
    assert not legs["alpha:sparse"].retrieve("절도", 5)


def test_hits_carry_the_corpus_qualified_retriever_name(tmp_path):
    make_corpus(tmp_path, "alpha", ["특허", "상표"])
    cfg = settings(tmp_path, "alpha", corpora=["alpha"])
    legs = {leg.name: leg for leg in build_corpus_legs(cfg)}
    hits = legs["alpha:sparse"].retrieve("특허", 5)
    assert hits and all(hit.retriever == "alpha:sparse" for hit in hits)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/legal_rag/unit/test_fanout.py -q`
Expected: FAIL — `ImportError: cannot import name 'build_corpus_legs'`

**Step 3: Write minimal implementation**

In `src/legal_rag/factory.py`, add below `build_hybrid_retriever`:

```python
def corpus_settings(settings: Settings, corpus_version: str) -> Settings:
    return settings.model_copy(
        update={"corpus": settings.corpus.model_copy(update={"version": corpus_version})}
    )


def build_corpus_legs(settings: Settings, *, embedder: Embedder | None = None) -> list[Retriever]:
    """Dense and sparse legs for every configured corpus, each named after its corpus.

    Each corpus keeps its own collection and its own BM25 statistics, so a term missing
    from one corpus's document frequencies simply yields nothing from that corpus rather
    than being dropped from the query everywhere.
    """
    embedder = embedder or build_embedder(settings)
    versions = settings.retrieval.corpora or [settings.corpus.version]
    legs: list[Retriever] = []
    for version in versions:
        scoped = corpus_settings(settings, version)
        check_index_compatibility(scoped, embedder_name=embedder.name)
        store = open_store(scoped, embedder)
        stats = BM25Stats.load(scoped.index_corpus_dir() / "bm25.json")
        dense = DenseRetriever(store, embedder)
        sparse = SparseRetriever(store, build_tokenizer_from_settings(scoped), stats)
        dense.name = f"{version}:dense"
        sparse.name = f"{version}:sparse"
        legs += [dense, sparse]
    return legs
```

`name` is a class attribute on both leg classes, but `retriever=self.name` resolves per
instance, so assigning it does not affect other instances.

Add `Retriever` to the existing `from legal_rag.retrieval import (...)` block if it is not
already imported; check before editing.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/legal_rag/unit/test_fanout.py -q`
Expected: PASS, 3 tests

**Step 5: Commit**

```bash
git add src/legal_rag/factory.py tests/legal_rag/unit/test_fanout.py
git commit -m "feat: build dense and sparse legs per corpus"
```

---

### Task 3: Reject corpora that disagree on the embedder

**Files:**
- Modify: `src/legal_rag/factory.py` (`build_corpus_legs`)
- Test: `tests/legal_rag/unit/test_fanout.py`

Dense scores compare only under one embedding model. Task 2 already passes
`embedder_name` to `check_index_compatibility`, so this may already pass — but the error
must name the offending corpus, or a four-corpus failure is unreadable.

**Step 1: Write the failing test**

Append to `tests/legal_rag/unit/test_fanout.py`:

```python
def test_a_corpus_indexed_with_another_embedder_is_named_in_the_error(tmp_path):
    from legal_rag.factory import IndexCompatibilityError
    make_corpus(tmp_path, "alpha", ["특허"])
    make_corpus(tmp_path, "beta", ["절도"])
    index = settings(tmp_path, "beta").index_corpus_dir() / "index.json"
    metadata = json.loads(index.read_text())
    metadata["embedder"] = "bge-m3:1024"
    index.write_text(json.dumps(metadata), encoding="utf-8")
    cfg = settings(tmp_path, "alpha", corpora=["alpha", "beta"])
    with pytest.raises(IndexCompatibilityError) as excinfo:
        build_corpus_legs(cfg)
    assert "beta" in str(excinfo.value) and "embedder" in str(excinfo.value)
```

**Step 2: Run it**

Run: `uv run pytest tests/legal_rag/unit/test_fanout.py -q`
Expected: it may already PASS, because `check_index_compatibility` reports the index
directory, which contains the corpus version. Read the assertion failure carefully before
changing anything.

**Step 3: Only if it fails, make the error name the corpus**

Wrap the per-corpus check in `build_corpus_legs`:

```python
        try:
            check_index_compatibility(scoped, embedder_name=embedder.name)
        except (IndexCompatibilityError, FileNotFoundError) as exc:
            raise IndexCompatibilityError(f"corpus {version!r}: {exc}") from exc
```

**Step 4: Confirm**

Run: `uv run pytest tests/legal_rag/unit/test_fanout.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/legal_rag/factory.py tests/legal_rag/unit/test_fanout.py
git commit -m "test: a corpus with a mismatched embedder is named in the error"
```

---

### Task 4: Weight each corpus in fusion

**Files:**
- Modify: `src/legal_rag/factory.py`
- Test: `tests/legal_rag/unit/test_fanout.py`

**Step 1: Write the failing test**

Append to `tests/legal_rag/unit/test_fanout.py`:

```python
def test_leg_weights_multiply_corpus_and_retriever_weights(tmp_path):
    from legal_rag.factory import fusion_weights
    cfg = settings(tmp_path, "alpha", corpora=["alpha", "beta"],
                   corpus_weights={"beta": 0.5}, dense_weight=1.0, sparse_weight=2.0)
    weights = fusion_weights(cfg)
    assert weights == {"alpha:dense": 1.0, "alpha:sparse": 2.0,
                       "beta:dense": 0.5, "beta:sparse": 1.0}


def test_a_down_weighted_corpus_ranks_below_a_full_weight_one(tmp_path):
    from legal_rag.retrieval.fusion import RRFFuser
    from legal_rag.schemas import Chunk, ChunkMetadata, DocumentType, ScoredChunk

    def hit(chunk_id, retriever):
        chunk = Chunk(id=chunk_id, document_id=chunk_id, document_type=DocumentType.statute,
                      path="p", order=0, text="t", title="t",
                      metadata=ChunkMetadata(dataset="d", corpus_version="v"))
        return ScoredChunk(chunk=chunk, score=1.0, retriever=retriever)

    fuser = RRFFuser(rrf_k=60, weights={"alpha:dense": 1.0, "beta:dense": 0.2})
    fused = fuser.fuse([[hit("beta-top", "beta:dense")],
                        [hit("alpha-1", "alpha:dense"), hit("alpha-2", "alpha:dense")]], k=3)
    assert [h.chunk.id for h in fused] == ["alpha-1", "alpha-2", "beta-top"]
```

The second test is the point of the whole design: without weights, `beta-top` at rank 1
would tie `alpha-1` and outrank `alpha-2`.

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/legal_rag/unit/test_fanout.py -q`
Expected: FAIL — `ImportError: cannot import name 'fusion_weights'`

**Step 3: Write minimal implementation**

In `src/legal_rag/factory.py`:

```python
def fusion_weights(settings: Settings) -> dict[str, float]:
    """Per-leg RRF weights: the corpus weight scales both of that corpus's legs."""
    versions = settings.retrieval.corpora or [settings.corpus.version]
    weights: dict[str, float] = {}
    for version in versions:
        corpus_weight = settings.retrieval.corpus_weights.get(version, 1.0)
        weights[f"{version}:dense"] = corpus_weight * settings.retrieval.dense_weight
        weights[f"{version}:sparse"] = corpus_weight * settings.retrieval.sparse_weight
    return weights
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/legal_rag/unit/test_fanout.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/legal_rag/factory.py tests/legal_rag/unit/test_fanout.py
git commit -m "feat: weight each corpus in rank fusion"
```

---

### Task 5: Dispatch so every caller gets fan-out unchanged

**Files:**
- Modify: `src/legal_rag/factory.py` (`build_hybrid_retriever`)
- Test: `tests/legal_rag/unit/test_fanout.py`

**Step 1: Write the failing test**

Append to `tests/legal_rag/unit/test_fanout.py`:

```python
def test_configured_corpora_produce_a_retriever_spanning_both(tmp_path):
    from legal_rag.factory import build_hybrid_retriever
    make_corpus(tmp_path, "alpha", ["특허", "상표"])
    make_corpus(tmp_path, "beta", ["절도", "사기"])
    cfg = settings(tmp_path, "alpha", corpora=["alpha", "beta"])
    retriever = build_hybrid_retriever(cfg)
    assert [leg.name for leg in retriever.legs] == [
        "alpha:dense", "alpha:sparse", "beta:dense", "beta:sparse"]
    hits = retriever.retrieve("절도", k=5)
    assert any(hit.chunk.id.startswith("beta") for hit in hits)


def test_no_configured_corpora_keeps_single_corpus_behaviour(tmp_path):
    from legal_rag.factory import build_hybrid_retriever
    make_corpus(tmp_path, "alpha", ["특허"])
    retriever = build_hybrid_retriever(settings(tmp_path, "alpha"))
    assert [leg.name for leg in retriever.legs] == ["alpha:dense", "alpha:sparse"]
```

Note the second test: leg names are now corpus-qualified even for a single corpus. That is
deliberate — one naming scheme, and the trace gains provenance. If any existing test
asserts the bare names `"dense"` or `"sparse"`, update it, and say so in the commit.

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/legal_rag/unit/test_fanout.py -q`
Expected: FAIL — legs are `["dense", "sparse"]`

**Step 3: Write minimal implementation**

Replace the body of `build_hybrid_retriever` so it uses the new helpers:

```python
def build_hybrid_retriever(settings: Settings, *, embedder: Embedder | None = None, store: VectorStore | None = None) -> HybridRetriever:
    embedder = embedder or build_embedder(settings)
    legs = build_corpus_legs(settings, embedder=embedder)
    return HybridRetriever(
        legs,
        RRFFuser(rrf_k=settings.retrieval.rrf_k, weights=fusion_weights(settings)),
        build_reranker(settings),
        k_leg=max(settings.retrieval.k_dense, settings.retrieval.k_sparse),
        k_fused=settings.retrieval.k_fused,
    )
```

The `store` parameter becomes unused. Verified before this plan was written: of the 15
references to `build_hybrid_retriever` across `src/`, `scripts/`, and `tests/`, none passes
`store=` — only the definition mentions it. Delete the parameter. It could not mean anything
when spanning corpora anyway.

**Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass. Existing retrieval tests exercise the single-corpus path through the
same code, which is the real regression check.

**Step 5: Commit**

```bash
git add src/legal_rag/factory.py tests/legal_rag/unit/test_fanout.py
git commit -m "feat: query every configured corpus from one retriever"
```

---

### Task 6: Fix the callers that assume bare leg names

**Files:**
- Modify: `scripts/eval_retrieval.py:34`
- Modify: `tests/legal_rag/integration/test_ingest_to_search.py:35-37`
- Test: `tests/legal_rag/unit/test_fanout.py`

Two places assume a leg is called exactly `dense` or `sparse`. Both were verified before
this plan was written; neither is hypothetical.

**Problem 1 — `--leg` stops working.** `scripts/eval_retrieval.py:34` reads:

```python
retriever = hybrid if leg == "hybrid" else next(l for l in hybrid.legs if l.name == leg)
```

With corpus-qualified names, `--leg sparse` matches nothing and raises `StopIteration`.
That flag is how you measure a single retrieval leg's Recall@k, which is exactly the
diagnostic this work needs. Under fan-out, `--leg sparse` should mean *all* sparse legs,
fused — not one corpus's.

**Problem 2 — leaked stores.** `tests/legal_rag/integration/test_ingest_to_search.py:35-37`
closes `retriever.legs[0].store` then breaks, which was right when both legs shared one
store. Fan-out gives one store per corpus, so the rest stay open and, in embedded mode,
keep their directory locks.

**Step 1: Write the failing test**

Append to `tests/legal_rag/unit/test_fanout.py`:

```python
def test_selecting_a_leg_kind_spans_every_corpus(tmp_path):
    from legal_rag.factory import build_hybrid_retriever, select_legs
    make_corpus(tmp_path, "alpha", ["특허"])
    make_corpus(tmp_path, "beta", ["절도"])
    cfg = settings(tmp_path, "alpha", corpora=["alpha", "beta"])
    retriever = build_hybrid_retriever(cfg)
    sparse = select_legs(retriever, "sparse")
    assert [leg.name for leg in sparse.legs] == ["alpha:sparse", "beta:sparse"]
    assert [leg.name for leg in select_legs(retriever, "hybrid").legs] == [
        leg.name for leg in retriever.legs]


def test_distinct_stores_are_closed_once_each(tmp_path):
    from legal_rag.factory import build_hybrid_retriever, close_legs
    make_corpus(tmp_path, "alpha", ["특허"])
    make_corpus(tmp_path, "beta", ["절도"])
    retriever = build_hybrid_retriever(settings(tmp_path, "alpha", corpora=["alpha", "beta"]))
    close_legs(retriever)  # must not raise on the second leg sharing a store
    assert True
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/legal_rag/unit/test_fanout.py -q`
Expected: FAIL — `ImportError: cannot import name 'select_legs'`

**Step 3: Write minimal implementation**

In `src/legal_rag/factory.py`:

```python
def select_legs(retriever: HybridRetriever, leg: str) -> HybridRetriever:
    """Narrow a retriever to one kind of leg across every corpus, for diagnostics."""
    if leg == "hybrid":
        return retriever
    chosen = [candidate for candidate in retriever.legs if candidate.name.rsplit(":", 1)[-1] == leg]
    if not chosen:
        raise ValueError(f"no {leg!r} leg; available: {[c.name for c in retriever.legs]}")
    return HybridRetriever(chosen, retriever.fuser, retriever.reranker,
                           k_leg=retriever.k_leg, k_fused=retriever.k_fused)


def close_legs(retriever: HybridRetriever) -> None:
    """Close each distinct store once; corpora share a store between their two legs."""
    for store in {id(leg.store): leg.store for leg in retriever.legs if hasattr(leg, "store")}.values():
        store.close()
```

Then in `scripts/eval_retrieval.py`, replace line 34 with:

```python
    retriever = select_legs(hybrid, leg)
```

and import `select_legs` from `legal_rag.factory`.

In `tests/legal_rag/integration/test_ingest_to_search.py`, replace the closing loop with:

```python
    close_legs(retriever)
```

**Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass.

Then confirm the flag still works end to end:

```bash
uv run python scripts/eval_retrieval.py --dataset data/eval/ip-sample-v1.jsonl \
  --corpus-version ip-sample-v1 --embedding-backend fake --leg sparse
```

Expected: a metrics table, not `StopIteration`.

**Step 5: Commit**

```bash
git add src/legal_rag/factory.py scripts/eval_retrieval.py \
  tests/legal_rag/integration/test_ingest_to_search.py tests/legal_rag/unit/test_fanout.py
git commit -m "fix: select and close legs by kind across corpora"
```

---

### Task 7: Prove fan-out on real corpora

**Files:**
- Test: `tests/legal_rag/test_fanout_real.py` (create), or run manually and record the output

The unit tests use synthetic vocabularies. This step runs the real thing.

**Step 1: Build small indexes for two real corpora**

`ip-sample-v1` already exists. Create a criminal sample beside it:

```bash
uv run python scripts/import_processed.py \
  --input-dir /home/layke/Documents/legal-rag/data/processed/criminal \
  --corpus-version criminal-sample-v1 --max-documents 40
uv run python scripts/build_index.py --corpus-version criminal-sample-v1 --embedding-backend fake
uv run python scripts/build_index.py --corpus-version ip-sample-v1 --embedding-backend fake
```

**Step 2: Fan out across both and confirm hits come from each**

```bash
LEGAL_RAG__RETRIEVAL__CORPORA='["ip-sample-v1","criminal-sample-v1"]' \
  uv run python scripts/search.py '특허 무효심판' \
  --corpus-version ip-sample-v1 --embedding-backend fake --k 10
```

Expected: results drawn from both corpora, and the stderr trace lines naming
`ip-sample-v1:dense`, `criminal-sample-v1:sparse`, and so on.

If the environment-variable form does not parse a JSON list, set `corpora` in a profile
YAML instead and pass `--profile`. Check `Settings.settings_customise_sources` for how
environment values are parsed before assuming either way.

**Step 3: Confirm a single corpus still behaves as before**

```bash
uv run python scripts/search.py '특허 무효심판' \
  --corpus-version ip-sample-v1 --embedding-backend fake --k 5
```

Expected: only `ip-sample-v1` hits, same as before this work.

**Step 4: Commit whatever you wrote down**

---

### Task 8: Run Qdrant as a server

**Files:**
- Create: `docker-compose.yml`
- Modify: `docs/legal-rag-backend.md`

Embedded Qdrant locks each index directory exclusively, so one process serving four corpora
blocks indexing and search on all four. Server mode also creates the payload indexes that
`VectorStore.recreate` skips when embedded.

**Step 1: Write the compose file**

```yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.12.1
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./data/qdrant:/qdrant/storage
    restart: unless-stopped
```

Pin the image tag. Add `data/qdrant/` to `.gitignore` if `data/**` does not already cover
it — check, because it probably does.

**Step 2: Start it and confirm**

```bash
docker compose up -d
curl -s http://localhost:6333/readyz
```

Expected: a ready response. `docker compose version` is v5.1.1 here, so use
`docker compose`, not `docker-compose`.

**Step 3: Commit**

```bash
git add docker-compose.yml
git commit -m "build: run Qdrant as a server for multi-corpus retrieval"
```

---

### Task 9: Index the production corpora into the server

This is the long one: roughly 497,000 chunks through bge-m3. Nothing in the code depends on
it, so run it when convenient.

**Step 1: Point at the server and index one corpus first**

```bash
export LEGAL_RAG__QDRANT__MODE=server
export LEGAL_RAG__QDRANT__URL=http://localhost:6333
uv run python scripts/build_index.py --corpus-version criminal-v1
```

Start with one corpus and confirm it completes before launching the rest. `bge-m3` is
already pulled in Ollama.

**Step 2: Index the remaining three**

Repeat for `civil-v1`, `administrative-v1`, and `ip-v1`.

**Step 3: Configure fan-out and measure**

Set in a profile YAML:

```yaml
retrieval:
  corpora: [criminal-v1, civil-v1, administrative-v1, ip-v1]
  k_fused: 50
```

`k_fused` defaults to 20, which is tight when eight lists feed the fuser rather than two.

Then measure, using the eval sets already in `data/eval/`:

```bash
uv run python scripts/eval_retrieval.py \
  --dataset data/eval/criminal-v1.jsonl --corpus-version criminal-v1 --json fanout.json
```

Compare against the same dataset with `corpora: []`. The difference between those two runs
is what fan-out actually costs and buys — Recall@k should hold or rise, precision may fall,
and that gap is the argument for building the cross-encoder reranker next.

---

## Out of scope

- The domain router: always-all plus weights was the chosen design.
- The cross-encoder reranker: `legal_rag.reranking.cross_encoder` does not exist and
  `sentence-transformers` is not installed. Build it after Task 9 shows what dilution costs.
- Index health states, blue/green publication, rebuild locking: these solve a corpus that
  mutates while serving. These four are imported in bulk and rebuilt into a new corpus
  version, which is already atomic because a new version means a new collection name.

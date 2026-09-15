# Multi-corpus retrieval fan-out

## Problem

Each legal corpus is its own Qdrant collection with its own BM25 statistics, because BM25
is corpus-global: `document_vector` bakes the corpus `avg_len` into every stored vector,
and `query_vector` keeps only terms present in that corpus's document-frequency map. That
isolation is correct and stays.

What is missing is the ability to answer one question from several corpora.
`build_hybrid_retriever` builds one retriever against one corpus version, so a query
reaches criminal or intellectual property, never both. Legal questions cross doctrinal
domains routinely.

## Design

### No new retriever class

`HybridRetriever` accepts `legs: list[Retriever]`, queries every leg, fuses all ranked
lists in a single pass, then reranks. That is already flat fan-out; it is merely handed
two legs from one corpus today.

Fan-out is therefore a factory change, not an architecture change. `Orchestrator`, the API,
`search.py`, and `eval_retrieval.py` all depend on the `Retriever` protocol and need no
modification.

```
build_fanout_retriever(settings)
  │
  ├─ for each corpus id in retrieval.corpora:
  │     settings.model_copy with corpus.version = id
  │     check_index_compatibility(per_corpus_settings, embedder_name=...)
  │     DenseRetriever  named "<corpus>:dense"
  │     SparseRetriever named "<corpus>:sparse"   (that corpus's own bm25.json)
  │
  └─ HybridRetriever(legs=<2 per corpus>, fuser=RRFFuser(weights), reranker, k_leg, k_fused)
```

`name` is a class attribute on both leg classes, but `retriever=self.name` resolves per
instance, so per-corpus naming needs no change to either class.

### Configuration

```yaml
retrieval:
  corpora: [criminal-v1, civil-v1, administrative-v1, ip-v1]
  corpus_weights: {ip-v1: 1.0}     # absent corpora default to 1.0
  k_fused: 50                      # raise this when fanning out
```

An empty `corpora` list preserves today's single-corpus behavior, so existing
configurations and tests are unaffected.

A leg's fusion weight is `corpus_weight × dense_weight` or `corpus_weight × sparse_weight`,
keyed on the leg name. `RRFFuser` already looks weights up by `hit.retriever`, unchanged.

`k_fused` deserves attention. With one corpus, two lists feed the fuser; with four, eight
do. Fusing 8 lists down to the current default of 20 before reranking discards candidates
that a reranker would otherwise rescue. Raise it in proportion to corpus count.

### Why routing is deferred

Reciprocal rank fusion weights every list equally: each corpus's rank-1 scores `1/(k+1)`
regardless of whether that corpus holds anything relevant. Fanning out to four corpora
therefore trades precision for recall, and something must repair it.

The chosen answer is weights plus a reranker rather than a router, because a weight
degrades gracefully where a hard gate loses the answer outright. A corpus weighted 0.3 still
contributes; a corpus a router excluded cannot. The reranker is the real precision gate and
is not implemented yet — `factory.build_reranker` imports `legal_rag.reranking.cross_encoder`,
which does not exist. Fan-out therefore ships with `rerank.backend: noop`, and the reranker
follows once Recall@k shows what dilution actually costs.

### Correctness rules

**Each corpus's sparse leg uses only its own statistics.** A term missing from criminal's
`df` makes criminal's sparse leg return nothing while intellectual property's leg, which
knows the term, returns hits. The failure mode of a shared collection becomes correct
behavior here, because statistics are never paired with a foreign collection.

**All corpora must share an embedder.** Dense scores compare only under one model, so every
corpus's `index.json` must report the same `embedder`. A mismatch fails loudly and names the
offending corpus, reusing the compatibility check already on the retrieval path.

**Deduplication stays keyed on `chunk.id`.** Civil and intellectual property share 636 chunk
ids, and inspection confirms these are the same judgments present in both AI Hub datasets:
identical text, identical `document_id`, differing only in `legal_domain`. Merging their
ranks is correct — a document two corpora both surface deserves to rank higher. The surviving
copy follows configured corpus order, so the reported domain is deterministic.

Note that the two upstream preprocessors use different id schemes: criminal and
administrative emit `<domain>::<source_key>::chunk_NNNN`, while civil and intellectual
property emit `<document_type>::<title>::<date>_chunk_NNNN`. The collisions arise from the
second scheme omitting the domain.

### Store lifecycle

Fan-out opens one `VectorStore` per corpus. Under Qdrant server mode these are HTTP clients
and cheap. Under embedded mode each holds an exclusive lock on its index directory, so one
process serving four corpora blocks indexing and search on all four. This is the existing
constraint multiplied by four, and it is why server mode is a prerequisite rather than a
later optimization.

## Prerequisite

Qdrant server mode, then re-index all four corpora into it — roughly 497,000 chunks against
bge-m3. Server mode also activates the payload indexes, which `VectorStore.recreate` skips
in embedded mode.

## Testing

- A term present in intellectual property's `df` but absent from criminal's still retrieves
  intellectual property hits, proving per-corpus statistics are used.
- A corpus whose `index.json` records a different embedder or tokenizer fails, naming it.
- Weighted fusion orders a down-weighted corpus's rank-1 below an up-weighted corpus's rank-2.
- A chunk present in two corpora appears once, with both contributions summed.
- An empty `corpora` list produces exactly today's single-corpus retriever.

## Out of scope

No domain router, no cross-encoder reranker yet, no index health-state machine, no
blue/green publication, no rebuild locking. The last three solve a corpus that mutates while
serving; these four are imported in bulk and rebuilt wholesale into a new corpus version,
which is already atomic because a new version means a new collection name.

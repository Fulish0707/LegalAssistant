# Structural civil retrieval pilot

The new corpus `civil-struct800-v1` retains the 75,624 civil-v1 documents and changes
only their chunking. Target 800 / maximum 1000 characters, with structural section
boundaries and sentence-window fallback. No overlap is introduced. There are
661,901 chunks (median 697 characters, p90 908, maximum 1000); 84.3% have parsed
structural headings. Character limits are not token limits: some passages still
exceed the reranker's 512 tokens. Size is a pilot choice, not a validated optimum.

## Build the separate index

Run from `/home/layke/Documents/MyLLMAgent`, with Qdrant and Ollama running:

```bash
LEGAL_RAG__CORPUS__VERSION=civil-struct800-v1 \
  .venv/bin/python scripts/build_index.py \
  --profile structural-civil --root /home/layke/Documents/MyLLMAgent \
  --corpus-version civil-struct800-v1 --on-disk
```

This builds both dense and BM25 vectors in a separate collection (with the default
embedder, `legal_civil_struct800_v1_ollama_bge_m3`). It does not index every corpus in
the profile's search fanout. The full corpus is already prepared; do not rechunk it
again under the same destination name. The new index has about 4.1 times as many
chunks as civil-v1, so budget extra indexing time, disk and BM25 memory. On-disk
vectors do not make BM25 fitting/loading memory-free. Run one heavy indexing/model
job at a time on the local GPU.

`build_index.py` recreates its selected collection. The version environment override
above pins the target even if the shell exported a different corpus version.
Re-running this command clears and rebuilds only that new collection; without `--resume` it starts from scratch. Existing civil-v1 remains the baseline.

## After indexing

Use `--profile structural-civil` in the backend CLI. To run a separate comparison
API without replacing the current instance:

```bash
.venv/bin/python scripts/serve.py --profile structural-civil --port 8001
```

The profile searches the new civil corpus and the existing criminal, administrative
and IP corpora. It holds local-profile retrieval depths, fusion weights and reranker
settings fixed. Environment overrides still take precedence over YAML; check any
exported `LEGAL_RAG__RETRIEVAL__CORPORA` before comparing.

After reranking, context assembly can add one preceding and one following chunk
from the same civil document/version. Ranked anchors take priority, duplicates are
removed, and every added passage has its own source ID and original legal header.
Neighbor evidence has score 0 and a `Context: Adjacent to S...` header: it has not
been independently ranked. Total rendered evidence, including headers and source
markers, is capped at 7500 characters. A neighbor that does not fit is omitted.
Expansion does not modify the retrieval ranking or reranker scores.

Generic defaults leave expansion disabled and retain the existing context policy.
An explicit `max_context_chars` also applies with radius zero, allowing a matched
budget when comparing expansion on/off. Expansion without an explicit budget uses
7500 characters. Pilot settings:

```yaml
rag:
  context_neighbor_radius: 1
  context_neighbor_corpora: [civil-struct800-v1]
  max_evidence_chars: 1500
  max_context_chars: 7500
```

Restricting expansion to the new civil corpus avoids unindexed document/order
lookups in old collections. New collections index document ID, corpus version and
chunk order. An empty neighbor-corpora list permits all versions when enabled.

## Compare before adopting

For a civil-only exact-gold retrieval comparison using the same 200 questions:

```bash
LEGAL_RAG__RETRIEVAL__CORPORA='["civil-v1"]' \
  .venv/bin/python scripts/eval_retrieval.py --profile local \
  --dataset data/eval/civil-pilot-200.jsonl \
  --json /tmp/civil-original-retrieval.json

LEGAL_RAG__RETRIEVAL__CORPORA='["civil-struct800-v1"]' \
  .venv/bin/python scripts/eval_retrieval.py --profile structural-civil \
  --dataset data/eval/civil-pilot-200.jsonl \
  --json /tmp/civil-structural-retrieval.json
```

Gold-document presence and rank do not establish legal sufficiency. Inspect whether
retrieved passages include the necessary rule, conditions and exceptions. Compare
chunking with expansion disabled first, then compare expansion on/off using identical
ranked hits. For answer evaluation, hold the model and rendered context budget fixed
in both arms, record the settings, and write to new report directories. Existing
saved Gemini answers used frozen old evidence and cannot evaluate this new index.

Do not reuse chunk-level relevance labels by ID: section IDs and boundaries changed.
Document-level gold labels remain comparable since documents are unchanged.

Indexing was completed by the user. Implementation checks now include 36 passing
focused tests and a live 200-question civil-only paired retrieval comparison.
Neighbor lookup/context assembly passed checks on 20 actual retrieval results.
A fresh paired answer pilot on 20 civil questions is also complete (see below).

## Resume an interrupted build

```bash
LEGAL_RAG__CORPUS__VERSION=civil-struct800-v1 \
  .venv/bin/python scripts/build_index.py --profile structural-civil \
  --corpus-version civil-struct800-v1 --on-disk --resume
```

Resume checks that the recomputed BM25 statistics match the original saved file,
checks collection configuration and stored chunk payloads, and embeds only missing
IDs. It never clears the collection. It scans from the beginning to find missing
points, so initial CPU work and increasing progress counters do not mean existing
embeddings are being regenerated. Keep the embedding model unchanged. If another
transport failure occurs, run the same resume command again. Completion metadata
is written only after all chunks are accounted for. Resume must not run alongside
another index writer targeting the same collection. Static review and CLI/syntax
checks succeeded; a live resume has not been run by the implementation agent.

## Measured retrieval comparison (2026-10-03)

Same 200 civil questions and retrieval settings, querying each complete civil index
separately with raw questions. Hybrid plus BGE reranking (512-token limit):

| Metric | civil-v1 | civil-struct800-v1 |
| --- | ---: | ---: |
| Gold Success@5 | 15.5% | 22.0% |
| Gold Success@10 | 20.0% | 24.5% |
| Gold Success@20 | 25.0% | 31.0% |
| MRR at retrieved depth | 0.1470 | 0.1985 |

Top-10 gain +4.5 percentage points: 20 cases improved, 11 regressed. Paired
bootstrap 95% interval [-1.0, +10.0] percentage points includes zero. This is a
promising development result, not a settled estimate of benefit. BM25-only top-10
was 22.5% to 23.5%. Structure and size changed together; their separate effects
are not identified. Document presence does not establish passage relevance.

On 20 live context checks, all ranked anchors survived, citations were distinct,
91 neighboring passages were added, and the maximum rendered evidence was 7478
characters against the 7500-character budget. This checks mechanics, not entailment.

Full multi-domain fanout and query rewriting remain unmeasured. Full per-case retrieval results are under
`data/eval/reports/structural-civil-test-2026-10-02/`, including `comparison.md`.

## Paired answer pilot (2026-10-03)

All 20 civil questions from the historical 80-case answer evaluation received fresh
answers for both arms: original civil-only retrieval and structural civil-only
retrieval with radius-1 neighbor expansion. Both used `gemini-3.7-flash`, temperature
0.2, low thinking, and a maximum rendered context of 7500 characters. References
were withheld from generation; complete answers were scored without truncation.
The same Qwen3:8B v2 grader used a 16384-token context for both arms.

| Automated metric | Original | Structural + neighbors |
| --- | ---: | ---: |
| Conclusion agreement | 14/20 (70%) | 14/20 (70%) |
| Reference agreement including basis | 14/20 (70%) | 14/20 (70%) |
| Grounded reference agreement | 14/20 (70%) | 14/20 (70%) |

Four cases improved and four regressed. The paired bootstrap 95% interval for
the score difference is [-25, +30] percentage points (10000 samples, seed 42).
All 40 grades resolved after retrying two duplicate-evidence-ID validation errors.
Maximum recorded prompt plus output was 12300 tokens; maximum rendered context
was 7498 characters.

This pilot shows no net automated answer-score gain. It is small, reuses development
questions, and combines chunking with context expansion. A review found concrete
judge explanation/rubric inconsistencies, so the labels need human calibration;
they do not establish legal correctness. Historical answer scores used different
evidence/settings and should not be treated as this experiment's baseline.

Review the eight changed cases and calibrate their grades before further broad
indexing or adoption. Saved answers, contexts, manifests, comparisons and calibration
notes are under `data/eval/reports/chat_pipeline/structural-answer-pair-2026-10-03/`.

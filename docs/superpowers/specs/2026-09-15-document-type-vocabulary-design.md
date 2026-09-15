# Document type vocabulary and citation category

## Problem

`document_type` serves two unrelated jobs, and conflating them corrupts evidence
rendering and citation verification.

As a **retrieval facet** it is fine-grained: six values the user filters on to reach
relevant material. As a **citation semantic** it is coarse: three classes that decide how
a chunk renders into evidence and what kind of citation it produces.

Two call sites need the coarse reading but test the fine one for equality:

- `rag/context.py:25` — `== DocumentType.court_decision` chooses the `Type: Case` header
  carrying Court, Case number, and Decision date. Everything else falls to an `else`
  branch that emits `Type: Interpretation` with Authority and Reference number.
- `verification/backfill.py:19` — the same equality test chooses `CitationKind.case`,
  otherwise `CitationKind.interpretation`.

The native normalizers emit `court_decision`. The upstream preprocessor emits `judgment`,
`decision`, `patent_trial_decision`, and `constitutional_decision`. None of those four
match, so every adjudicative chunk from the upstream corpora renders as an interpretation
with two empty fields and verifies as the wrong citation kind — roughly 340,000 chunks
across the four domains. Nothing raises; retrieval and grounding quietly degrade.

## Ground truth

Measured from `/home/layke/Documents/legal-rag/data/processed/*/chunks.jsonl`:

| Type | civil | criminal | administrative | intellectual_property |
|---|---|---|---|---|
| `statute` | ✓ | ✓ | ✓ | ✓ |
| `judgment` | ✓ | ✓ | `judgement` | ✓ |
| `legal_interpretation` | ✓ | ✓ | ✓ | ✓ |
| `constitutional_decision` | — | ✓ | ✓ | — |
| `decision` | ✓ | — | — | ✓ |
| `patent_trial_decision` | — | — | — | ✓ |

All six are already members of `DocumentType`. The enum needs no addition.

The four-type vocabulary in the file-structure specification (§5) is incomplete: it omits
`decision` and `patent_trial_decision`, about 70,000 documents in civil and IP.

Administrative spells the type `judgement` in 46,016 chunks. `DocumentType("judgement")`
raises `ValueError`, so administrative import fails on its first such record.

## Design

Separate the two axes. Keep `document_type` as the stored, filterable retrieval facet.
Derive a three-value citation category from it.

```
document_type (6, stored)                 citation_category (3, derived)
─────────────────────────────────────     ──────────────────────────────
statute ──────────────────────────────►   statute
judgment ─────────────────────────────┐
decision ─────────────────────────────┤
patent_trial_decision ────────────────┼►  case
constitutional_decision ──────────────┤
court_decision (native) ──────────────┘
legal_interpretation ─────────────────┐
admin_interpretation (native) ────────┴►  interpretation
```

`constitutional_decision`, `decision`, and `patent_trial_decision` map to `case` because
each is adjudicative and carries an authority and a case number.

One mapping in `schemas/documents.py` is the single source of truth, feeding three call
sites:

1. `rag/context.py` — branch on category to choose the evidence header.
2. `verification/backfill.py` — branch on category to choose the `CitationKind`.
3. `schemas/query.py` and `retrieval/filters.py` — a new `citation_category` field on
   `RetrievalFilters` expands to `MatchAny("document_type", [...member types])`.

The category is never stored. Server-side filtering still works, because the filter
expands to the member types already present in the Qdrant payload. No frozen model
changes, no re-import dependency, and no risk of a stored category drifting from its type.

Native `court_decision` and `admin_interpretation` keep working unchanged. The
`Literal`-pinned discriminators on `CourtDecision` and `AdminInterpretation`, the
`LegalDocument` discriminated union, `make_document_id` output, and the `isinstance`
dispatch in `chunking/router.py` are all untouched.

### Rejected alternatives

**Collapse the vocabulary** — retire `court_decision` and `admin_interpretation`, one
spelling per concept. Rejected because flattening `patent_trial_decision` into `judgment`
destroys the granularity IP relevance depends on; 68,135 of 76,142 IP documents are patent
trial decisions. It also breaks the discriminated union, document IDs, and every native
test.

**Store a `citation_category` field on `ChunkMetadata`** — identical behavior at higher
cost: a frozen model change, a re-import to populate, and a value that can drift from the
type it describes.

**An anonymous set predicate** — fixes the two branches but yields no named concept to
filter or route answers by, so it does not serve category separation at retrieval time.

### The `judgement` typo

Fix upstream and regenerate administrative. `import_processed` documents itself as
validating "without reading QA or changing sources", and it records a SHA-256 of each
source file in `validation_report.json`. An alias map inside the importer would silently
rewrite source data and leave the recorded hash describing something other than the
corpus. The upstream repository is on disk, so regeneration is available.

## Testing

- A test asserting every `DocumentType` member maps to a category, so a new member cannot
  be added without classifying it. This is the regression guard for the original bug.
- `context.py` renders `Type: Case` with Court and Case number for each case-like type.
- `backfill.py` returns `CitationKind.case` for each case-like type.
- A `citation_category` filter expands to a `MatchAny` over exactly the member types.
- Importing a chunk whose `document_type` is unknown fails with `file:line` context.

## Risks

Administrative reports `qa_validation_issues: 30593` against `qa_pairs: 42228`, and
criminal `30593` against `41907` — roughly 70% of QA records flagged upstream. This does
not block import, because `import_processed` never reads `qa.jsonl`, but it does block the
document-level Recall@K evaluation those files exist to support. Investigate before
trusting retrieval metrics.

`source_key` is present only in criminal and administrative, the two external-source
domains. Any source-level deletion command is therefore inherently two-of-four domains.

This design does not address the corpus-global BM25 statistics problem, which gates the
shared `legal_chunks` collection. `BM25Stats.fit_stream` computes IDF and average document
length over a whole corpus version, so ingesting domain by domain into one collection
would store sparse vectors computed against different statistics. That decision remains
open and must be resolved before adopting a single collection.

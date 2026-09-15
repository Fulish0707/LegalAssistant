# Document Type Vocabulary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Stop ~340,000 imported chunks from rendering as the wrong evidence type and verifying as the wrong citation kind, by deriving a coarse citation kind from the fine-grained `document_type` instead of testing equality against one native spelling.

**Architecture:** `document_type` stays the stored, filterable retrieval facet with six values. A single mapping function in `schemas/documents.py` derives `CitationKind` from it. Three call sites consume that mapping: the evidence header builder, the citation backfill, and a new retrieval filter that expands a citation kind into a `MatchAny` over its member document types. The derived value is never stored, so no frozen model changes and no re-import dependency.

**Tech Stack:** Python 3.11, Pydantic v2, pytest, Qdrant, uv.

**Design doc:** `docs/superpowers/specs/2026-09-15-document-type-vocabulary-design.md`

**Refinement from the design:** the design called the derived value `citation_category`. `CitationKind` already exists in `schemas/answer.py` with exactly `statute` / `case` / `interpretation`, and `answer.py` imports nothing from the package, so there is no circular-import risk. Reuse `CitationKind` rather than adding a parallel enum that must be kept in sync.

**Run tests with `uv run pytest`.** The project is not importable from the anaconda python on this machine.

---

### Task 1: Derive CitationKind from DocumentType

**Files:**
- Modify: `src/legal_rag/schemas/answer.py` (append after `CitationKind`)
- Modify: `src/legal_rag/schemas/__init__.py` (export)
- Test: `tests/legal_rag/unit/test_schemas_documents.py`

**Where this lives, and why.** The mapping goes in `answer.py`, not `documents.py`. Every module in `schemas/` imports downward toward `documents.py` (`chunks.py`, `evidence.py`, and `query.py` all do). `answer.py` already owns `CitationKind` and currently imports nothing from the package, so `from .documents import DocumentType` runs with that direction. Putting the mapping in `documents.py` would make the base module depend on the answer contract — no runtime cycle, but backwards layering.

**Step 1: Write the failing test**

Append to `tests/legal_rag/unit/test_schemas_documents.py`:

```python
def test_citation_kind_covers_every_document_type():
    # Regression guard: a new DocumentType member must be classified, not silently
    # fall through to "interpretation" the way judgment/decision once did.
    for document_type in DocumentType:
        assert isinstance(citation_kind(document_type), CitationKind)


def test_citation_kind_groups_adjudicative_types_as_case():
    case_like = {
        DocumentType.judgment,
        DocumentType.court_decision,
        DocumentType.decision,
        DocumentType.patent_trial_decision,
        DocumentType.constitutional_decision,
    }
    for document_type in case_like:
        assert citation_kind(document_type) is CitationKind.case
    assert citation_kind(DocumentType.statute) is CitationKind.statute
    assert citation_kind(DocumentType.legal_interpretation) is CitationKind.interpretation
    assert citation_kind(DocumentType.admin_interpretation) is CitationKind.interpretation
```

Add to that file's imports:

```python
from legal_rag.schemas import CitationKind, DocumentType, citation_kind
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/legal_rag/unit/test_schemas_documents.py -q`
Expected: FAIL with `ImportError: cannot import name 'citation_kind'`

**Step 3: Write minimal implementation**

Append to `src/legal_rag/schemas/answer.py`, below the `CitationKind` class:

```python
# Retrieval filters on the fine-grained document_type; evidence rendering and citation
# verification need only the coarse kind. Deriving one from the other keeps a new
# document type from silently classifying as an interpretation.
_CITATION_KINDS: dict[DocumentType, CitationKind] = {
    DocumentType.statute: CitationKind.statute,
    DocumentType.judgment: CitationKind.case,
    DocumentType.court_decision: CitationKind.case,
    DocumentType.decision: CitationKind.case,
    DocumentType.patent_trial_decision: CitationKind.case,
    DocumentType.constitutional_decision: CitationKind.case,
    DocumentType.legal_interpretation: CitationKind.interpretation,
    DocumentType.admin_interpretation: CitationKind.interpretation,
}


def citation_kind(document_type: DocumentType) -> CitationKind:
    return _CITATION_KINDS[document_type]


def document_types_for(kind: CitationKind) -> list[DocumentType]:
    return [dt for dt, mapped in _CITATION_KINDS.items() if mapped is kind]
```

Put `from .documents import DocumentType` with the other imports at the top of `answer.py`, not inline.

In `src/legal_rag/schemas/__init__.py`, add `citation_kind` and `document_types_for` to the `from .answer import (...)` line and to `__all__`, keeping alphabetical order.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/legal_rag/unit/test_schemas_documents.py -q`
Expected: PASS

Then confirm the package still imports cleanly, since this task moves an import between sibling modules:

Run: `uv run python -c "import legal_rag.schemas"`
Expected: no output, exit 0

**Step 5: Commit**

```bash
git add src/legal_rag/schemas/answer.py src/legal_rag/schemas/__init__.py tests/legal_rag/unit/test_schemas_documents.py
git commit -m "feat: derive CitationKind from DocumentType"
```

---

### Task 2: Branch evidence headers on citation kind

**Files:**
- Modify: `src/legal_rag/rag/context.py:19-34`
- Test: `tests/legal_rag/unit/test_rewriter_context.py`

**Step 1: Write the failing test**

Append to `tests/legal_rag/unit/test_rewriter_context.py`:

```python
def test_context_builder_renders_imported_judgment_as_case():
    # An imported judgment carries the same court/case metadata as a native
    # court_decision and must not fall through to the interpretation header.
    chunk = Chunk(
        id="d9#p", document_id="d9", document_type=DocumentType.judgment, path="p", order=0,
        text="피고인은 무죄", title="t",
        metadata=ChunkMetadata(dataset="f", corpus_version="v", court="인천지방법원",
                               case_number="2008보5", decision_date=date(2008, 7, 14)),
    )
    bundle = ContextBuilder().build([ScoredChunk(chunk=chunk, score=1.0, retriever="reranked")])
    header = bundle.items[0].header
    assert header["Type"] == "Case"
    assert header["Court"] == "인천지방법원" and header["Case number"] == "2008보5"
    assert "Authority" not in header
```

Add to that file's imports:

```python
from datetime import date

from legal_rag.schemas import Chunk, ChunkMetadata
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/legal_rag/unit/test_rewriter_context.py::test_context_builder_renders_imported_judgment_as_case -q`
Expected: FAIL — `assert 'Interpretation' == 'Case'`

**Step 3: Write minimal implementation**

In `src/legal_rag/rag/context.py`, change the import line to:

```python
from legal_rag.schemas import CitationKind, Evidence, EvidenceBundle, ScoredChunk, citation_kind
```

Before dropping `DocumentType` from that import, confirm nothing else in the file uses it:

Run: `grep -n DocumentType src/legal_rag/rag/context.py`
Expected: only the import line. If any other line appears, keep `DocumentType` in the import.

Replace the two branch conditions (lines 19 and 25):

```python
            kind = citation_kind(chunk.document_type)
            if kind is CitationKind.statute:
                header = {
```

```python
            elif kind is CitationKind.case:
                header = {
```

Leave the `else` interpretation branch and everything below it unchanged.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/legal_rag/unit/test_rewriter_context.py -q`
Expected: PASS, 6 tests. The pre-existing `test_context_builder_assigns_ids_and_headers` must still pass — it uses native `court_decision` and proves the change is backward compatible.

**Step 5: Commit**

```bash
git add src/legal_rag/rag/context.py tests/legal_rag/unit/test_rewriter_context.py
git commit -m "fix: render imported judgments as case evidence"
```

---

### Task 3: Classify citations by citation kind

**Files:**
- Modify: `src/legal_rag/verification/backfill.py:16-21`
- Test: `tests/legal_rag/unit/test_verification.py`

**Step 1: Write the failing test**

Append to `tests/legal_rag/unit/test_verification.py`:

```python
def test_backfilled_citation_kind_follows_document_type():
    for document_type in (DocumentType.judgment, DocumentType.patent_trial_decision,
                          DocumentType.constitutional_decision, DocumentType.decision):
        evidence = Evidence(source_id="S1", chunk_id="c", document_id="d",
                            document_type=document_type, header={"Type": "Case"},
                            text="대법원 2018다12345 판결", score=1.0)
        assert _kind(evidence) is CitationKind.case
```

Add `_kind` to that file's imports:

```python
from legal_rag.verification.backfill import _kind
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/legal_rag/unit/test_verification.py::test_backfilled_citation_kind_follows_document_type -q`
Expected: FAIL — `assert <CitationKind.interpretation> is <CitationKind.case>`

**Step 3: Write minimal implementation**

In `src/legal_rag/verification/backfill.py`, replace the whole `_kind` function:

```python
def _kind(evidence: Evidence) -> CitationKind:
    return citation_kind(evidence.document_type)
```

Update the import line to bring in `citation_kind` and drop `DocumentType` if it is now unused in the file (check with `grep -n DocumentType src/legal_rag/verification/backfill.py` before removing).

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/legal_rag/unit/test_verification.py -q`
Expected: PASS, all pre-existing verification tests included.

**Step 5: Commit**

```bash
git add src/legal_rag/verification/backfill.py tests/legal_rag/unit/test_verification.py
git commit -m "fix: classify citations by derived citation kind"
```

---

### Task 4: Filter retrieval by citation kind

**Files:**
- Modify: `src/legal_rag/schemas/query.py:28-38` (`RetrievalFilters`)
- Modify: `src/legal_rag/retrieval/filters.py:31-35` (`to_conditions`)
- Test: `tests/legal_rag/unit/test_schemas_misc.py`

This is what makes category separation usable at query time: a caller asks for `case` and Qdrant filters server-side, with no new payload field.

Adding a field changes what `is_empty()` reports, since it compares against a default instance. Verified before writing this plan: `is_empty()` has no callers in `src/`, so nothing downstream changes behavior. Re-check with `grep -rn "is_empty" src/` if that looks stale.

**Step 1: Write the failing test**

Append to `tests/legal_rag/unit/test_schemas_misc.py`:

```python
def test_citation_kind_filter_expands_to_document_types():
    conditions = to_conditions(RetrievalFilters(citation_kind=CitationKind.case))
    assert len(conditions) == 1
    assert conditions[0].key == "document_type"
    assert set(conditions[0].values) == {
        "judgment", "court_decision", "decision", "patent_trial_decision", "constitutional_decision",
    }


def test_explicit_document_types_win_over_citation_kind():
    # A narrower explicit list must not be widened back out to the whole category.
    filters = RetrievalFilters(document_types=[DocumentType.patent_trial_decision],
                               citation_kind=CitationKind.case)
    conditions = to_conditions(filters)
    assert [c.values for c in conditions if c.key == "document_type"] == [["patent_trial_decision"]]


def test_empty_filters_still_detected():
    assert RetrievalFilters().is_empty()
    assert not RetrievalFilters(citation_kind=CitationKind.case).is_empty()
```

Add the imports that file needs:

```python
from legal_rag.retrieval.filters import to_conditions
from legal_rag.schemas import CitationKind, DocumentType, RetrievalFilters
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/legal_rag/unit/test_schemas_misc.py -q`
Expected: FAIL — `ValidationError: Extra inputs are not permitted [citation_kind]` (the model is `extra="forbid"`)

**Step 3: Write minimal implementation**

In `src/legal_rag/schemas/query.py`, add the field to `RetrievalFilters` directly below `document_types`:

```python
    citation_kind: CitationKind | None = None  # coarse facet; expands to document_types
```

and add `from .answer import CitationKind` to that file's imports.

In `src/legal_rag/retrieval/filters.py`, replace the `document_types` clause in `to_conditions`:

```python
    if filters.document_types:
        conditions.append(MatchAny("document_type", [dt.value for dt in filters.document_types]))
    elif filters.citation_kind:
        conditions.append(MatchAny("document_type", [dt.value for dt in document_types_for(filters.citation_kind)]))
```

and import `document_types_for`:

```python
from legal_rag.schemas import RetrievalFilters, document_types_for
```

An explicit `document_types` list wins, so a caller can narrow within a category.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/legal_rag/unit/test_schemas_misc.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/legal_rag/schemas/query.py src/legal_rag/retrieval/filters.py tests/legal_rag/unit/test_schemas_misc.py
git commit -m "feat: filter retrieval by citation kind"
```

---

### Task 5: Lock in the import-time failure for unknown types

**Files:**
- Test: `tests/legal_rag/test_import_processed.py`

No implementation change. `DocumentType(...)` already raises `ValueError` and `import_processed` already wraps row errors with `file:line`. This test documents that contract so nobody later "fixes" the administrative typo with a silent alias map.

**Step 1: Write the test**

Read the existing fixture helpers at the top of `tests/legal_rag/test_import_processed.py` and reuse them rather than writing new ones. Add a test that writes a chunk row whose `document_type` is `"judgement"`, runs `import_processed`, and asserts:

```python
    with pytest.raises(ValueError) as excinfo:
        import_processed(input_dir, settings)
    assert "judgement" in str(excinfo.value)
    assert "chunks.jsonl:1" in str(excinfo.value)
```

**Step 2: Run it**

Run: `uv run pytest tests/legal_rag/test_import_processed.py -q`
Expected: PASS immediately. If it fails, the `file:line` wrapping is weaker than assumed — fix the assertion to match actual behavior rather than changing the importer.

**Step 3: Commit**

```bash
git add tests/legal_rag/test_import_processed.py
git commit -m "test: unknown document_type fails import with file and line"
```

---

### Task 6: Full suite and review

**Step 1: Run everything**

Run: `uv run pytest -q`
Expected: all pass. Pay attention to `test_store.py` and `test_ingestion_normalize.py` — they exercise the native `court_decision` path that must still work.

**Step 2: Confirm no stray equality tests remain**

Run: `grep -rn "DocumentType.court_decision\|DocumentType.admin_interpretation" src/`
Expected: only `schemas/documents.py` (the mapping and the `Literal` discriminators) and `ingestion/normalize/*.py` (`make_document_id` calls). Any `==` comparison left in `rag/` or `verification/` is a missed call site.

**Step 3: Commit if anything changed**

---

### Task 7: Fix the `judgement` spelling upstream

**Files:**
- Modify: the preprocessor in `/home/layke/Documents/legal-rag/` (locate with `grep -rn "judgement" /home/layke/Documents/legal-rag/src/`)

This is a separate repository. Do not start it until Tasks 1-6 are committed.

**Step 1: Find the source of the spelling**

Run: `grep -rn "judgement" /home/layke/Documents/legal-rag/ --include=*.py`
Expected: the administrative document type mapping, likely a `DocuType` code table where `02` maps to `judgement`.

**Step 2: Fix and verify no other domain is affected**

Correct it to `judgment`. Then run `grep -rn "judgement" /home/layke/Documents/legal-rag/ --include=*.py` again and expect no hits.

**Step 3: Regenerate administrative only**

Re-run that repo's preprocessing for the administrative domain. Do not regenerate the other three — their outputs are already correct and criminal/IP hashes are recorded in this repo's history.

**Step 4: Verify the output**

```bash
grep -o '"document_type": *"[a-z_]*"' \
  /home/layke/Documents/legal-rag/data/processed/administrative/chunks.jsonl \
  | sort | uniq -c
```

Expected: `judgment` with 46,016 occurrences, and no `judgement`. Counts for `statute` (14,004), `legal_interpretation` (7,449), and `constitutional_decision` (19,255) should be unchanged.

---

### Task 8: Re-import all four domains

Both previously imported corpora were deleted on 2026-09-15, so this is a clean import, not a migration.

**Step 1: Import each domain to its own corpus version**

```bash
uv run python scripts/import_processed.py \
    --input-dir /home/layke/Documents/legal-rag/data/processed/criminal \
    --corpus-version criminal-v1
```

Repeat for `civil`, `administrative`, and `intellectual_property`. Run `--validate-only` first on each if you want the report without writing.

**Step 2: Confirm the vocabulary landed**

Each corpus writes `validation_report.json` with a `document_types` map. Confirm administrative shows `judgment` and no `judgement`, and that no domain reports a type outside the six.

**Step 3: Prove the fix on real data**

Everything up to here verifies the mapping in isolation. This step is the only one that proves the original 340,000-chunk bug is actually gone.

`ip-sample-v1` survived the 2026-09-15 deletion and its index is still at `data/indexes/ip-sample-v1`. It predates this work and carries the same `patent_trial_decision` and `decision` types, so it is a valid smoke test and small enough to rebuild in seconds.

```bash
uv run python scripts/build_index.py --corpus-version ip-sample-v1 --embedding-backend fake
uv run python scripts/search.py --corpus-version ip-sample-v1 --embedding-backend fake "특허 무효심판"
```

Expected: at least one hit whose `document_type` is `patent_trial_decision`, rendering with `Type: Case` and a `Court` or `Authority` line — not `Type: Interpretation` with empty fields. Check `scripts/search.py --help` first; its flags may differ from `build_index.py`.

If the evidence header still reads `Interpretation` for a `patent_trial_decision` hit, stop. The unit tests pass but a call site was missed — go back to Task 6 Step 2.

**Step 4: Note what is not done here**

Do not build indexes yet. `BM25Stats.fit_stream` computes corpus-global statistics, so how these four corpora relate to one Qdrant collection is an open design question recorded in the design doc's risk section. Indexing per corpus version works today and is unaffected by this plan.

---

## Out of scope

- The shared `legal_chunks` collection and the corpus-global BM25 problem.
- `source_key` on `ChunkMetadata`, which exists only in criminal and administrative.
- The ~70% upstream `qa_validation_issues` rate, which blocks Recall@K evaluation but not import.

# Graded relevance evaluation with an LLM judge

## Problem

Exact-gold retrieval metrics cannot measure this system. Each processed QA row was
generated from one document, but its question states a general legal rule: across all
76,142 intellectual-property QA rows, no question contains a case number or the document
title. Hundreds of decisions state the same rule, so the one gold document is rarely the
only correct answer.

The 200-case ip-v1 pilot shows the result. Only 21 of 200 golds appear within the top 100
in any mode. Inspection of the stored hybrid failures finds the top hits usually do answer
the question — the 상표법 non-use clause for a cancellation-trial question, a 대법원
definition of 부정경쟁의 목적 — while the gold is an unrelated-looking decision about a
different mark. In 33 of 50 failures, hybrid's top 3 share more vocabulary with the
reference answer than the gold chunks do.

Tuning fusion weights or adding a reranker against these labels would fit noise: a
two-case change moves Recall@20 by one point.

## Goal

Replace "is the gold document retrieved" with "are retrieved chunks relevant to the
question", judged by a local LLM, validated against human labels, and extendable as new
retrieval configurations surface chunks nobody has judged.

Scope: the existing 200-case ip-v1 target (`target.jsonl`, sha256 prefix `8e3196`), with
the `legal_domain=intellectual_property` filter. Under that filter the criminal, civil and
administrative legs return nothing, so this set does **not** measure multi-corpus fan-out.

## Design

### 1. Judgment store — persistent and append-only

`data/eval/relevance/ip-v1-pilot-200/judgments.jsonl`, one row per judged pair:

```json
{"case_id": "qa:…", "chunk_id": "…", "document_id": "…", "grade": 2,
 "reason": "…", "judge_model": "ollama:qwen3:8b", "prompt_sha256": "…",
 "judged_at": "2026-09-17T…Z", "raw": null}
```

The key is `(case_id, chunk_id, judge_model, prompt_sha256)`. A judgment is never rewritten;
changing the model or prompt makes new keys, so old and new judgments coexist and never mix
silently. `store.json` records the target hash, filters, grade definitions and prompt text.

Why append-only rather than a frozen label file: the reranker, the main planned consumer,
exists to promote chunks from ranks 21–100 into the top 10. A pool frozen at top 20 would
count those promotions as unjudged, and scoring unjudged as irrelevant would penalize the
reranker for exactly the behaviour it is meant to have.

### 2. Pooling

`scripts/build_relevance_pool.py` retrieves the top 20 chunks for every case in dense,
sparse and hybrid mode (same settings and filter as the diagnostic runs) and writes the
union to `pool.jsonl`: case, question, reference answer, chunk id, document id, type,
title, full chunk text, and each mode's rank. Expected size is about 7–8k pairs.

Chunks are the judging unit. Document relevance is the maximum grade over its judged
chunks; the reverse derivation is impossible.

Known gold chunks for every case are added to the pool whatever their rank, so the judge
can be checked against them.

### 3. Judge

`scripts/judge_relevance.py` grades pool rows missing from the store for the configured
judge key. It is resumable: each judgment is appended and flushed as it completes.

- Model `qwen3:8b` via `OllamaChatModel`, `think=False`, `temperature=0`, JSON schema
  `{grade: 0|1|2, reason: str}` (prompt v1; see "Calibration v1 result and prompt v2").
- The prompt shows the question, the reference answer, and the chunk (type, title, text).
  It never reveals the mode, rank, or whether the chunk belongs to the gold document.
- Grades are defined against the reference answer:
  - **2** — the chunk states the legal rule or finding the reference answer relies on,
    sufficient to support that answer alone. Any decision stating the rule qualifies,
    not only the gold.
  - **1** — the same doctrine or provision, but insufficient to support the answer alone.
  - **0** — unrelated, or only superficially shares terms.
- A `StructuredOutputError` is retried once. A second failure is stored with
  `grade: null` and the raw text, and excluded from metrics with a reported count. A parse
  failure is never coerced to 0.

`--judge-missing --run <dir>` tops up pairs that a new retrieval run surfaced within its
evaluated depth, so later configurations extend the same store.

The reference answer was written from the gold document, so gold-phrased chunks may be
favoured slightly. The report states this.

### 4. Human calibration

`scripts/sample_calibration.py` draws 50 pairs **before** any LLM judgment is shown,
stratified so the sample is not almost all grade 0:

| Stratum | Pairs |
|---|---|
| gold chunks | 10 |
| best rank 1–3 in any mode | 15 |
| best rank 4–10 | 15 |
| best rank 11–20 | 10 |

It writes `calibration.csv`. The human fills it in without seeing LLM grades, so every
row leads with what is needed to find the source quickly:

| Column | Example |
|---|---|
| `case_number` | `2016노4825`, `2022기정1485`, `09-0381`, `상표법 제73조` |
| `authority` | `서울중앙지방법원`, `특허심판원`, `법제처` |
| `date`, `title`, `document_type`, `document_id` | as stored |
| `search_query` | `서울중앙지방법원 2016노4825` — paste into a legal search site |

then `question`, `reference_answer`, `chunk_text`, and the empty `human_grade` and
`human_note`.

The corpus leaves `metadata.case_number` empty, so the number comes from `document_id`:
a `<court>-<number>` judgment splits on its first hyphen, a 법제처 interpretation keeps
its trailing `NN-NNNN` 안건번호, a statute becomes the law name plus the `제N조` that
opens the chunk text, and anything else (patent-trial and 공정위 decisions) uses the
`document_id` unchanged. A label that cannot be derived falls back to `document_id`
rather than being left blank.

`scripts/judge_agreement.py` joins human and LLM grades and reports:

- 3×3 confusion matrix;
- linearly weighted Cohen's kappa;
- binary agreement, precision and recall on **grade 2 vs {0, 1}**, the boundary that
  drives the headline metric;
- judge grade on the gold chunks. Many gold chunks graded 0 means the prompt is broken.

The full run should not be trusted unless the binary kappa on grade 2 is at least 0.6.
Below that, revise the prompt, re-judge only the 50, and repeat; a new prompt hash keeps
the attempts apart.

#### Calibration v1 result and prompt v2

Calibration failed for every prompt tried (`qwen3:8b`, think off). Three rubric wordings are
recorded in `judge_runs.jsonl`, identified by prompt hash; binary grade-2 kappa did not move:
`2ac3175d19db` 0.34, `73296afdcb2b` 0.35, `b733e2805288` 0.31. The `agreement.json` of the last
(`b733e2805288`) gives precision 0.42 and exact agreement 0.50: the judge gave 2 to 19 of the 50
pairs, the reference labels to 12, and 25 pairs disagree. The reference labels were
produced by the model "GPT 6 Astra", not by a person. Of the 10 gold chunks, the reference
graded only 1 as 2 (6 as 0, 3 as 1), so a gold chunk is often not an answer.

Reading all 25 disagreements of `b733e2805288` showed four patterns:

- The judge quoted the reference answer's wording as if the chunk said it, and over-graded to 2.
- The same field but a different issue, procedure or provision was graded relevant (a 보정각하
  case for 정정청구 기각사유, 저작권 침해 for 특허권 침해, 제42조 제3항 for 제4항).
- A rule stated in different words was missed, and under-graded.
- Reference answers sometimes cite the wrong article, so anchoring grades on the answer
  propagates its errors.

Because rewording the rubric did not help, prompt v2 adds a mechanical check. It grades against the question's legal issue and treats the reference answer as a
fallible hint. Grade 0 explicitly covers a different issue, institution, procedure or right.
The schema becomes `{evidence_quote, grade, reason}`, quote first. Code verifies the quote
against the chunk as shown (whitespace collapsed, surrounding quote marks and ellipses
stripped, at least 15 characters) and stores an unverified grade 2 as 1, keeping the model
grade, quote and verification result in `details`. Think is a run parameter in the key, with
512 output tokens off and 4,096 on, `num_ctx` 8192 for both. `judge_agreement.py` writes
per-key `agreement-<hash12>.json` files that record `reference_source`, `quote_downgrades` and
agreement before the quote check, so the effect of verification can be seen.

Prompts are identified by hash prefix, not version number. The prompt text of
`b733e2805288` (called "v3" in its own notes) was never committed, so it is not in git history.

### 5. Metrics

`src/legal_rag/evaluation/graded_metrics.py`; the binary functions in
`retrieval_metrics.py` stay unchanged.

For a ranked list of chunks with judgments:

- **nDCG@10** with gain `2^grade − 1` and an ideal ranking of all judged chunks for the
  case sorted by grade. This is the headline metric.
- **Success@5/10/20** — at least one grade-2 chunk in the top k.
- **MRR** — reciprocal rank of the first grade-2 chunk, within 100.
- **Precision@10** at grade ≥ 1 and at grade 2.
- **judged_coverage@10** — the fraction of top-10 chunks that are judged. A comparison is
  valid only when this is 1.0 for every run; otherwise `--judge-missing` must run first.

Each metric is reported at chunk level and at document level, where a document takes the
rank of its first chunk.

Pooled recall is omitted: its denominator depends on pool contents and cannot be compared
across pool versions.

`scripts/eval_graded.py` takes run directories (or runs retrieval itself), computes these
metrics from the store, and writes the same three-file layout as the mode comparison:
`comparison.md`, `comparison.json` and `cases.csv`.

## Order of work

1. Pool (minutes, embeddings on GPU).
2. Sample calibration and label 50 pairs (human, 1–2 hours).
3. Judge the same 50 and check agreement; iterate on the prompt if needed.
4. Judge the full pool overnight (~7–8k calls on qwen3:8b).
5. Compare dense, sparse and hybrid with graded metrics.
6. Only then tune weights or add a reranker, topping up judgments for each new run.

## Testing

- Store: appends never overwrite; judgments with a different model or prompt hash are
  distinct; resuming skips judged keys.
- Judge: a fake chat model returns grades; a malformed response retries once and then
  stores `null`, never 0; the prompt excludes rank, mode and gold status.
- Metrics: graded nDCG against hand-computed values; unjudged chunks lower coverage and
  count as grade 0 only inside the reported coverage figure; document-level ranks use
  each document's first chunk.
- Calibration: the sampler meets stratum quotas deterministically for a seed; agreement
  on a hand-built confusion matrix matches known kappa values.

## Out of scope

Unfiltered, fan-out pools; other corpora; answer-generation quality; automatic failure
categorisation; the reranker itself.

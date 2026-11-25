# Excerpt-selector test: design and pass bar, written before any selector was scored (2026-09-29)

## Question

With the current top-10 ranking fixed, can a selector that reads no labels recover the verified grade-2 quote
visibility that the label-aware oracles in `../live-flask-current-2026-09-29-rootvenv/priority1-evidence-budget-report.md`
say is available inside Flask's evidence budget (five documents, 1,200 characters each, 6,000 in all)?

The evidence-budget report is the reference: same 200 IP + 60 mixed-domain development questions, same captured top 10
(`flask_current_k100`), same formatter, same quote matching (judge verifier normalization, credit only inside the quote's
own document), same existing Qwen judgments. Nothing is judged again and no serving setting changes. Its
`make_case_views`, `variants_for_case` and `quote_occurrences` are imported, not re-implemented.

## Gate (before any selector is read)

The harness must reproduce the report's baseline and all six oracle rows exactly: IP 460/1089 quotes and 171/195 cases,
mixed 119/244 and 48/56, and the oracle rows in `priority1-evidence-budget-analysis.json`.

## Selectors

Every selector chooses at most 1,200 characters of each shown document's `joined` text (distinct chunk texts of the
document's top-10 hits in rank order, joined by newline-ellipsis-newline). A document of at most 1,200 characters is shown
whole. The display span is the chosen window widened evenly on both sides to 1,200 characters, kept inside the document
(`windows.display_span`). Candidate windows (`windows.py`) are sentence-packed, never cross a chunk separator, and are as
long as the reranker can read whole: query plus window at most 512 tokens under the model's own tokenizer, measured. They
overlap by about half, and the window at character 0 is always a candidate. No window size, stride or margin is tuned.

| Name | Documents shown | Window chosen by |
| --- | --- | --- |
| `prefix` | first five (Flask today) | first 1,200 characters |
| `ce_window_top5` (a) | first five | highest cross-encoder score of (query, window) |
| `lex_window_top5` (b) | first five | highest IDF-weighted query-token coverage (RegexTokenizer; document frequency over the dataset's pool chunks) |
| `ce_docs_prefix` (c0) | five highest by best cross-encoder window score | first 1,200 characters |
| `ce_docs_window` (c) | five highest by best cross-encoder window score, shown in that order | highest cross-encoder window score |
| `lex_docs_window` | five highest by best lexical window score | highest lexical window score |

Ties go to the earlier window or document. The cross-encoder is `BAAI/bge-reranker-v2-m3`, fp16, CUDA, max length 512,
batch 16, built as serving builds it, and scores windows of all ten top-10 documents.

## Metrics

Per dataset, over questions that have a verified grade-2 quote in the top 10 (195 IP, 56 mixed): visible quotes (pairs)
and questions with at least one visible quote, as in the report. Against `prefix`: questions and quotes gained, lost and
net. Share of the oracle gap closed, net new questions divided by the oracle's new questions: `ce_window_top5` and
`lex_window_top5` against the same-top-5 character-window oracle (IP 16, mixed 5), `ce_docs_prefix` against the best-5
prefix oracle (16 and 2), `ce_docs_window` and `lex_docs_window` against the best-5 character-window oracle (24 and 8).
Also the mean number of shown documents (of five) that hold a visible verified quote. Cost: windows scored and wall time per
question.

## Pass bar

* `ce_window_top5` passes if, on both datasets, it gains net at least half the oracle gap in questions (IP at least 8,
  mixed at least 3), loses at most one question that has a visible quote today, and has a positive net change in quotes.
* `ce_docs_window` passes if, on both datasets, it gains net at least half the best-5 window gap (IP at least 12, mixed at
  least 4) and loses at most one such question.
* The lexical rows, `ce_docs_prefix` and `lex_docs_window` have no bar. They show whether the cross-encoder is needed and
  which lever carries the gain.

A pass means the selector deserves an answer-quality check, not adoption. A miss on both ends the window idea for this
budget.

## What this cannot show

Each pair has one Qwen-chosen quote. A selector that shows a different passage that also answers the question scores as a
miss, so results are a floor for the selector. The measure is quote retention, not answer quality, and not Recall@k. Human
support labels and independent judge calibration do not exist. The two datasets are the development sets the oracles were
built on, so nothing here is confirmatory. Prompt size stays at or under 6,000 characters in every row: a larger budget is
not tested (on the `/v1/chat` path, `k_final` 8 raised gold coverage but not answer accuracy, `docs/legal-rag-backend.md`
on the main branch).

## Erratum (added after the results; the text above is unchanged)

The last sentence of the previous section says that on the `/v1/chat` path `k_final` 8 "raised gold coverage but not answer
accuracy". That overstates what the answer test showed. The coverage gain was measured on the 210-question formal set
(0.471 to 0.543), but in the 80-question answer sample the gold document entered the evidence for 48 questions against 47,
so that test never had more coverage to convert into accuracy. It says nothing about whether better evidence improves
answers. This test does not depend on that sentence.

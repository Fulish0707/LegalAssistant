# Four levers after the test-set repair (2026-10-09)

Four independent pieces of work, one subagent each. Each produces a measurement and a recommendation. **None changes
what is served**: no edit to tracked files, to `configs/`, to `.env`, to prompts under `src/`, or to any corpus or
Qdrant collection. A change to the product is proposed in the result and decided by the user afterwards.

## Shared rules (every subagent)

- **Work folder.** `data/eval/reports/chat_pipeline/four-levers-2026-10-09/<n>-<name>/`. Write only there. Scripts go
  there too (`scripts/` is git-ignored and `data/` is not tracked, so nothing you write is committed).
- **Method.** Write `PREREG.txt` before any run: the question, the data, the arms, the measure, and the rule that
  decides what you will recommend. Write `RESULT.txt` at the end. If you change the plan after seeing a result, add a
  dated amendment that says it is post hoc. Report failures and dropped steps plainly.
- **One GPU.** The card is 8 GB and holds one heavy job. Every command that loads the chat model, the grader, the
  reranker or builds a retriever MUST be run as
  `data/eval/reports/chat_pipeline/four-levers-2026-10-09/gpu.sh <command>` from the repository root. It waits for a
  lock, so a command can sit for an hour or more before it starts: start it in the background and poll its output
  file; do not start a second copy. Never call Ollama or load a model outside `gpu.sh`. CPU-only analysis of stored
  files needs no lock.
- **Memory.** Host RAM is tight and `earlyoom` kills Qdrant when free memory falls under about 3 GB. Do not load
  whole corpora into Python; stream files. Qdrant runs at `localhost:6333` (do not restart it, do not start Docker
  Desktop). Never use `pkill -f` with a pattern that appears in your own command line.
- **Python.** `.venv/bin/python` from the repository root for everything under `src/legal_rag` and `scripts/`.
  Before running: `unset LEGAL_RAG__PATHS__ROOT`.
- **Test sets.**
  - Held-out, repaired: `data/eval/testset-v2/` (`README.txt`, `items.jsonl` with 213 questions, `kind` statute | other,
    `gold_articles`, `core`). Grade with
    `scripts/judge_answers.py judge --run <dir> --conclusions data/eval/testset-v2/core_conclusions.json`.
  - Stored served retrieval and answers for those questions: `final-test-2026-10-06/{formal,everyday}.frozen.jsonl`
    (five hits per question, full chunk text, reranker score), `.../{formal,everyday}/verify.jsonl` (answers,
    evidence as shown) under `data/eval/reports/chat_pipeline/`.
  - Development: the 80 questions frozen in `prompt-tuning-2026-10-06/frozen.jsonl`. Tune on development data only.
    The held-out set is touched once, at the end, with the rule already written.
- **Known measurement limits (use them, do not rediscover them).**
  - The core answer grader (qwen3:8b) gives a different grade for 5 to 9 of 213 identical answer-and-core pairs on a
    second pass, and identical generation arms differ by 3 to 6 of 80. A difference inside that is noise; run each arm
    twice and report strict and lenient where you grade answers.
  - Gold-document coverage does not measure retrieval for general (civil) questions. Judge-free retrieval measures that
    work: for `kind == statute`, an article of the gold law among the five shown passages (rule in
    `final-test-2026-10-06/report.py`, function `covered`); for the 24 `other` questions with `gold_articles`, the
    same rule.
  - The passage grader is lenient on on-topic passages; a grader-only "supported" needs a second read.
  - Report counts beside rates, and distinct source questions (the hex id after `qa:`) beside question counts; formal
    and everyday wordings of one question share that id.
- **Hand-back.** Your final message is read by the coordinating session, not by the user: give the result first, the
  numbers with their denominators, what you recommend and what you could not do. Plain statements, no padding.

---

## 1. Everyday wording and the reranker (`1-rewrite/`)

**What is known.** `short-article-check-2026-10-09/` : for 54 statute questions that exist in formal and everyday
wording, the served reranker scores the gold article 0.99 (formal) against 0.80 (everyday), difference -0.19,
95% CI [-0.28, -0.12]; 10 everyday pairs fall under 0.5 and no formal one. When the score is under 0.5 the article is
among the five shown passages for 6 of 12 everyday questions, against 33 of 35 at 0.9 or more.

**What the code does (verified).** `src/legal_rag/rag/rewriter.py` can ask the chat model for one or two legal-term
search phrasings (`rag.llm_rewrite`, off; prompt `src/legal_rag/llm/prompts/rewrite_system.md`). They are added as
extra *search* queries. But `Orchestrator.handle` calls
`retriever.retrieve_many(queries.search_queries, k, filters, rerank_query=question)`: the reranker always scores
against the user's original wording. So turning `llm_rewrite` on can bring the article into the pool but cannot
raise its reranker score. The lever to test is therefore the *rerank query*, not only the setting.

**Arms** (retrieval only, no answer generation; same code path as serving except where stated):
- A. served: stored frozen hits (no run needed; check a sample of 20 reproduces).
- B. `rag.llm_rewrite = true`, rerank by the original question (the existing setting as it is).
- C. as B, and rerank by the first rewritten phrasing.
- D. as B, and rerank by both the original and the rewritten phrasing, each candidate keeping its higher score.
C and D need no product edit: build the retriever with `legal_rag.factory.build_hybrid_retriever`, call
`retrieve_many` yourself with the queries and the `rerank_query` you want (for D, two rerank passes over the same
fused list, taken from `retriever.last_trace["fused"]` or by calling the reranker directly). The rewrites come from
one pass of the chat model over the questions (qwen3:4b, temperature 0, the prompt file unchanged), saved to a file and
reused by B, C and D so that the arms differ only in how the rewrite is used.

**Data.** Development first: the everyday statute questions are held-out, so develop on the 80 development questions
(does the pipeline run, are rewrites sane, does anything break), then run the four arms once on all 213 held-out
questions.

**Measures.** (1) Judge-free statute coverage in the five shown passages, everyday and formal apart (served baseline
from the stored hits; recompute it with your own code first and confirm it matches 110 of 115 on version 2).
(2) The same for the 24 `other` questions with `gold_articles`. (3) For `other` questions without a gold article: how
many of the five shown chunks change against A (a retrieval change you cannot score is still a cost; report it).
(4) Reranker score of the gold article per arm for the 54 pairs. (5) Added latency per question (rewrite call, extra
legs, second rerank pass). Paired comparison per arm against A with gained/lost counts and a bootstrap interval over
distinct source questions.

**Decision rule to register.** Recommend an arm only if everyday statute coverage gains with an interval that excludes
0, formal statute coverage and the 24 `other` questions lose at most 2 each, and you state the latency. Otherwise:
not shown. Also read by eye at least 20 rewrites: a rewrite that adds a law name the question did not contain, or
changes the meaning, is a defect to count.

**Out of scope.** Changing the rewrite prompt (note what you would change, do not tune it on held-out data);
generating answers.

---

## 2. Answers lean on the first passage (`2-first-passage/`)

**What is known.** `caselaw-evidence-2026-10-07/generation-read.jsonl`: of 22 answers that disagreed with the
reference although a shown passage stated the conclusion, the supporting passage was not ranked first in 14, and in
about 9 or 10 of those the answer built its conclusion from passage 1 (one reader, not blind). The answer prompt is
`src/legal_rag/llm/prompts/answer_system.md`; answers are structured (`summary`, `analysis`, `citations` with
`source_id` S1..S5, ...).

**First, measure the behaviour without a grader** (CPU only, stored data). The stored runs keep the answer text
(`verify.jsonl`: `final_text`, first 1,500 characters) and the five passages as shown (`evidence`), but NOT the
structured `citations` with their `source_id`s (checked 2026-10-09). So attribute by text: for each answer, the
overlap between the answer and each of the five passages (for example character 4-gram recall of the answer against
the passage, plus whether the passage's law name with article or its case number appears in the answer), and call
the passage with the highest overlap the one the answer drew on. Validate the attribution before using it: on the
22 answers of `generation-read.jsonl` the reader's notes say which passage the answer followed; report how often your
attribution agrees, and stop if it is poor. Then report: share of answers that draw on passage 1; the distribution
over ranks; and, for the held-out questions where `caselaw-evidence-2026-10-07/support.jsonl` marks which shown
passage states the conclusion (`group == "main"`, `n` = passage index from 0, `stated and not contradicts`), how often
the answer draws on a supporting passage, split by that passage's rank, and how agreement with the reference differs.
A lean exists only if answers draw on passage 1 more often than its share of supporting passages explains. If the
measurement does not show it, say so and stop before any prompt work.
For new runs you may also record the cited `source_id`s: wrap the run in a script of your own that patches the
generator's `generate` to log `answer.citations` (the way `eval_chat_pipeline.py verify --answer-prompt` patches
`load_prompt`); do not edit `scripts/` or `src/`.

**Then the prompt arm** (only if the measurement shows the lean). Write one variant of the answer prompt as a file in
your folder (never edit the file under `src/`): the same text plus an instruction, in Korean and in the style of the
existing rules, to read every [LEGAL SOURCE] block before answering, to base the conclusion on the block that states
the rule the question asks about and not on the first block, and to say so when the blocks differ. Keep everything
else byte-identical; check that with a diff. One variant, written before any run; do not iterate on wording.

**Runs.** `scripts/eval_chat_pipeline.py verify --frozen <frozen.jsonl> --answer-prompt <file>` with
`--overrides '{"llm":{"backend":"ollama","model":"qwen3:4b","max_tokens":2048}}'` (this is how
`prompt-format-2026-10-07/run.sh` did it; read that script and reuse its structure, including the strict grading
step). Development set, each arm twice, order base, variant, variant, base. About 40 minutes of GPU per arm.

**Measures.** The attribution measures above per arm (and cited source ids if you record them); conclusion agreement lenient and strict; verifier counts
(first draft flagged, stripped). **Rule to register:** the variant is recommended for a held-out confirmation only if
the share of answers drawing on a passage other than the first moves by more than the difference between the two runs of the same
arm AND conclusion agreement does not fall by more than 3 of 80 under either wording. If it passes, run base and
variant once each on the held-out frozen evidence (version 2 questions only, graded with the version 2 conclusions)
and report the same measures plus the "cites the supporting passage" rate.

**Out of scope.** Changing retrieval order or the number of passages; editing the served prompt.

---

## 3. Outdated statute text in the older corpora (`3-stale-statutes/`)

**What is known.** `statute-reference-review-2026-10-08/RESULT.txt`: the corpus snapshot the test references were
written from is behind the current law; for 19 of 126 statute questions the current text changes the answer (mostly
2025 agency renames, four of substance). The served retrieval searches five corpora: `criminal-v1`, `civil-v1`,
`administrative-v1`, `ip-v1` (which contain statutes as old multi-article windows, chunk ids like
`administrative::statute::...` or court-style ids with `document_type == "statute"`) and `statute-api-full-v1`
(current law, one article per chunk, ids `statute:<hash>#art:...`).

**Question.** How often does the assistant show, and repeat, outdated statute text, and what would removing the old
statute windows from retrieval cost?

**Step 1, CPU only, stored data.** For the 274 held-out questions (`final-test-2026-10-06/*.frozen.jsonl`, all of
them, not only version 2): how many of the five shown passages are statute chunks from the four older corpora against
`statute-api-full-v1`; for how many questions an old statute window is shown; for how many the old window and a
current article of the same law are both shown. Then staleness: the agency renames are listed in
`data/eval/testset-v2/build.py` (`RENAMED`); count shown old-corpus statute chunks that contain an old agency name,
and stored answers (`verify.jsonl`, `final_text`) that use an old name where the current law has the new one. For
substance, use the 39 "TEXT DIFFERS" articles of `statute-reference-review-2026-10-08/packets.jsonl`: was the old
version of such an article shown, and did the answer follow the old text? Read the answers concerned by eye; there
will be few.

**Step 2, retrieval only (reranker, through `gpu.sh`).** Arms on all 274 questions: A served (stored); B the same
retrieval with statute chunks of the four older corpora removed from the fused candidates before reranking (do it in
your script around `retrieve_many`: take the fused list from the trace or filter the pool at `k = k_fused`, then
keep the first five; do not edit the product). Confirm first that arm A reproduced by your script equals the stored
top five on at least 95% of questions; if not, report drift and stop.
Measures: judge-free statute coverage (version 2 questions, formal and everyday apart); the 24 `other` questions with
`gold_articles`; for the rest, how many shown chunks change; how many questions lose a shown passage that the
passage reader had confirmed as stating the conclusion (`caselaw-stage-loss-2026-10-07/confirmed.jsonl` and
`caselaw-evidence-2026-10-07/support.jsonl` for the served five).

**Rule to register.** Recommend removing the old statute windows only if statute coverage loses at most 2 questions per
wording, the 24 lose at most 1, and step 1 shows the old windows are actually shown and repeated (give the counts).
Say how the removal could be done in the product (a retrieval filter on corpus and document type against re-indexing
the four corpora without statutes) and what each would touch; do not do it.

**Out of scope.** Deleting or rebuilding any collection; re-fetching law.

---

## 4. A floor for weak evidence (`4-weak-evidence/`)

**What is known.** On the 148 non-statute held-out questions, no shown passage scored 0.5 or more with the reranker for
28, and answers there agreed with the reference 12 of 28 times, against 62 of 92 when all five did. The assistant
answers in every case; nothing tells the reader the sources were weak. An earlier note records answers written from
evidence scored 0.07 to 0.36.

**This is analysis and design, mostly CPU.** No change to the product.

**Step 1.** On development data (the 80 frozen questions with their stored graded runs under
`prompt-tuning-2026-10-06/` and `prompt-format-2026-10-07/`): relate reranker scores of the five shown passages to
answer agreement. Candidate signals: top score, count at or above 0.5, mean of the top three. For each, the curve of
"answers flagged" against "share of flagged answers that disagree with the reference" and "share of all wrong answers
caught". Choose ONE signal and ONE threshold on development data and write them into PREREG.txt before step 2.

**Step 2.** Apply that fixed rule once to the held-out version 2 questions (stored hits and the grades in
`testset-v2-baseline-2026-10-09/`), statute and other apart: how many answers would be flagged, how many of those are
wrong, how many wrong answers are missed. Compare with two trivial rules (flag nothing; flag at random the same
number) so the reader can see what the signal adds. Because grades are noisy, also give the counts using the grades
of `reference-repair-2026-10-08/rescore.jsonl` and `statute-reference-review-2026-10-08/rescore.jsonl` (lenient,
version 2) where they differ.

**Step 3.** Read by eye 20 flagged answers (10 the grader calls right, 10 wrong): is the evidence really off the
question, and what would a reader have needed to hear? From that, write up to three concrete behaviours with the exact
Korean wording each would show: (a) a notice above the answer that the sources found are only loosely related;
(b) answering with the general rule and asking one clarifying question; (c) declining to state a conclusion. For each:
where in the code it would be decided (the Flask app already appends notices after the answer in `app.py` around the
call to `annotate_unsupported`; the service has a clarification path in `src/legal_rag/rag/clarify.py`), what it
would cost on answers that were right, and how it would be tested.

**Rule to register.** Recommend a floor only if, on held-out, at least half of the flagged answers are wrong under both
sets of grades and the rule flags no more than a quarter of all answers. Otherwise say the signal is too weak to act
on and give the numbers.

**GPU.** None expected. If you find you need reranker scores that are not stored, use `gpu.sh`.

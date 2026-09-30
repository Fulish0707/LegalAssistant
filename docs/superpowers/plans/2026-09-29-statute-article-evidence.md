# Article-Level Statute Evidence Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `/v1/chat` show the statute article that answers the question, instead of the first 1,500 characters of a statute window, without re-embedding anything.

**Architecture:** After `retrieve_many` returns the cross-encoder-ranked candidates (depth `k_fused`), a new `ArticleEvidenceSelector` takes the first 30 fused candidates plus the reranker's own top 10, splits each statute window at article headings, scores every article against the question with the cross-encoder that is already loaded, and returns the best `k_final` units as chunks whose text is one article and whose `metadata.article` names it. `ContextBuilder`, the generator and the verifier are unchanged. The step is off by default (`rag.article_evidence`).

**Tech Stack:** Python 3.11, pydantic-settings, sentence-transformers CrossEncoder (`BAAI/bge-reranker-v2-m3`, fp16, already served), pytest, typer scripts, Ollama (qwen3:4b answers, qwen3:8b judge), Qdrant.

---

## Why this change

The 80-question `/v1/chat` answer run of 2026-09-28 left 30 answers not correct. Reading all 30 (single rater, `data/eval/reports/chat_pipeline/formal-2026-09-28-articles/failure-breakdown/`) put 15 down to questions that name no specific law or case, 9 to evidence that lacks the key passage (7 of them statute questions), 5 to the model and 1 to the judge. This plan covers the statute part of those 9, which is measurable and fixable without new data.

Statute corpora are cut into windows of about 3,450 characters that hold several articles. The context builder shows the first 1,500 characters of each window. On the 35 statute questions the gold statute is shown for 24 (69%), but the article the reference cites is shown for 13 by a heading match (16 by the acceptance script's rule, which also counts the article a window opens in). Answers were correct for 10 of 11 when the cited article was shown and 12 of 24 when it was not.

The replay below reruns the analyzer, rule rewriting and retrieval with the served settings (top-5 of the replay matched the run's evidence in 31 of 35 questions, 97.7% of items) and follows the key chunk, the window that holds the cited article, through every stage:

| Where the cited article is lost (22 questions where it is not shown) | Questions | Evidence |
| --- | ---: | --- |
| Key chunk is in the top 5, but the article lies past the 1,500-character cut | 2 | display cut |
| Key chunk is a fused candidate, but the cross-encoder ranks it 6 to 100 | 13 | It reads about 512 tokens (~900 characters) of a 3,450-character window. Where the cross-encoder kept the chunk in the top 5 the cited heading sits a median 389 characters in; where it demoted it to rank 21 or lower, a median 1,489 (898 to 2,625) |
| Key chunk is not among the 100 fused candidates | 7 | 3 are in no leg's top 100 (questions that name no law), 4 are in one leg's top 100 at ranks 10 to 60 and were cut by fusion |

By fused order alone the key chunk is in the top 5 for 20 of 35 questions; by the served cross-encoder order, for 15. When the key chunk is a candidate at all (28 of 35), scoring each article of the fused candidates with the same cross-encoder ranks the cited article first in 28 of 28.

Pool choice (offline replay on the 35 statute questions and 45 other questions; "extra pairs" are cross-encoder pairs on top of today's 100):

| Candidates whose statute windows are split | Cited article shown (of 35) | Gold document shown, other questions (of 45; today 23) | Extra pairs, statute questions |
| --- | ---: | ---: | ---: |
| Today (no step) | 13 | 23 | 0 |
| First 30 by fused rank | 26 | 19 | 151 |
| First 50 by fused rank | 27 | 21 | 242 |
| All 100 | 28 | 23 | 413 |
| First 30 by fused rank plus the reranker's top 10 (chosen) | 27 | 23 | 169 |

A fused-only pool loses other questions because the reranker promotes chunks from beyond the pool; adding the reranker's own top 10 removes that loss.

The implementation was run against the live index on all 80 questions (`scripts/eval_article_evidence.py`, Task 6): statute questions 16 to 28 covered (+12, -0), other questions 23 to 23, step time 1.27 s (p90 1.98 s) and 169 extra pairs on statute questions, 0.21 s and 20 pairs on the others. Evidence text on statute questions shrinks from 7,342 to 3,537 characters.

### Tested and not needed

- **Larger `k_final`.** At 8 chunks, 1 of the 6 key chunks appeared and its sentence was past the cut; statute answers went from 22 to 19 correct.
- **Deeper legs or a larger `k_fused`.** `k_fused` 150 makes 2 more key chunks candidates (28 to 30 of 35); a leg floor of 5 or 10 makes none.
- **Neighbour chunks.** Adding the previous and next window of each shown gold window includes the key chunk for 20 of 35 questions but shows the article for only 15, because it usually lies past the cut.
- **Article-level re-chunking and re-embedding.** Article scoring already ranks the cited article first whenever its window is a candidate, so the index does not need to change.

### Limits

Seven statute questions stay uncovered. For three the key chunk is in no leg's top 100 ("세액공제를 신청하려면…", "대통령령으로 정한 관계는…", "후발적 사유로 인정되는 경우는…"), for three it is in one leg's top 100 only, at ranks 10 to 31, and fusion cut it ("적립된 순이익금은…", "건축물의 건축 허가를 받은 후…", "시ㆍ도지사는 어떤 권한을…"), and for one it is a fused candidate at rank 66 ("새마을금고와 관련한 결산보고서…"). The category labels come from one reader and the key chunk is found by heading match, so treat the counts as good to a question or two. The answer effect is estimated, not measured. By the acceptance script's rule, 15 of 16 statute questions were answered correctly when the cited article was shown and 7 of 19 when it was not. Eight of the 12 gained questions are wrong today, so the ceiling is about +7 correct out of 35 (8 x 0.94). The two groups also differ in how specific their questions are, so plan for +3 to +7. A text-level check on the 12 gained questions supports the mechanism: the judge's reference point lies inside the shown text for 11 of 12 with the step (median bigram overlap 0.86, against 0.57 today), so the gain is in the text and not only in the article label. Task 7 measures the answer effect.

**All of the numbers above are in-sample.** The 80 questions are formal AIHub questions, written from the gold document, so a question often paraphrases the article it asks about, and the pool size (30 plus 10) was chosen on them. Task 6b runs the same step, pool unchanged, on 169 formal questions outside the 80 and on the everyday-phrased set. Result (2026-09-29): on the 169 formal questions the cited article is shown for 35 of the 82 statute questions today and 69 with the step (+36, -2), and on the 150 everyday questions for 30 of 73 today and 54 with the step (+27, -3); the other questions do not change in either set, and the statute step time p90 is 1.77 s and 1.72 s. The gain holds (41% and 33% of the statute questions against 34% in-sample), but the everyday set loses 3 against the bar of at most 2, so criterion 3 is not met (Task 6b, Step 4).

**Behaviour change beyond showing the right article.** Evidence for a statute question shrinks from 7,342 to 3,537 characters, so prompts get shorter. If a longer context turns out to help, raising `retrieval.k_final` with the flag on is a separate experiment (`--overrides '{"rag": {"article_evidence": true}, "retrieval": {"k_final": 8}}'`); this plan does not include it.

**Known limits of the splitter and the selector** (from the code reviews of Tasks 1 to 5; measured on all 21,234 statute windows of the four corpora unless noted).

- The splitter loses no text (the segments of every window reproduce it) and never cuts a line-start cross-reference as a heading. Two first windows carry a wrongly labelled opening because the article locator falls back to the first article (`administrative::statute::011102` and `004631`), and a deleted article written as `제N조` with `삭제` on the next line (189 lines in 23 documents) stays attached to the article before it.
- Adjacent windows overlap by about 400 characters, so an article on a window edge appears as two fragments and the step keeps the better-scoring one. For 34% of adjacent pairs neither fragment contains the other (median fragment 750 characters, merged article 1,254), so the kept unit can be half an article; a lexical simulation puts the loss at about 6.6% of answer lines on such pairs. The measured gains include this.
- 2.0% of article segments exceed the 1,500-character display cut (0.85% exceed 2,000), and the cross-encoder reads only the first ~880 characters of any of them.
- Units drop the neighbouring articles that the old 1,500-character window happened to show, so a draft that cites such a neighbour now fails verification (`statute_not_in_evidence`). This feeds the first-draft-flagged bar of criterion 4.
- Chunks that carry different paragraphs of one article are kept apart (the four imported corpora have none). Keep `article_keep_top` at `k_final` or above: the reranker's own top hits outside the fused pool would otherwise be dropped.
- The step adds a cross-encoder call of about 170 pairs on the request thread. `predict` is not serialised, so overlapping chat requests interleave GPU batches; peak VRAM and latency with two simultaneous chats are not measured. An exception in the step fails the request, as one in the reranker does.
- With `rag.article_evidence: true` and `rerank.backend: noop` the server starts and `/health` says ok, and every `/v1/chat` returns a 500; the reason (`rag.article_evidence needs rerank.backend: cross_encoder`) is only in the server log.
- Only `/v1/chat` and the CLI use the step. The Flask chat app reaches this backend through `/v1/search` only, so its users get no article evidence.
- With the flag on, `trace.retrieved["reranked"]` lists all `k_fused` reranked candidates instead of `k_final`, and `evidence_ids` holds unit ids such as `w1@1`. `HybridRetriever.last_trace` is per thread, so a reader on another thread sees `{}`.
- The three new settings change `Settings.hash()` of every profile, so `mode_comparison` refuses to compare a run recorded before this change with one recorded after. Nothing else keys on the hash.

## Acceptance criteria

Criterion 2 was written after the in-sample harness numbers existed, on the same 80 questions the pool size was chosen on, so it is a regression guard and not evidence. Criterion 3 (bars written before the held-out runs, `data/eval/reports/chat_pipeline/formal-2026-09-28-articles/statute-replay/heldout-bars.txt`) and criterion 4 are the real tests.

1. **Tests.** `538 passed, 3 skipped` before the change, `562 passed, 3 skipped` after (24 new tests).
2. **Retrieval, in-sample (Task 6, 80 questions).** Statute questions covered with the step at least 26 of 35, none lost, at least 8 gained. Other questions covered at least 22 of 45 (23 today). Mean step time at most 2.0 s on statute questions and 0.5 s on the others. Mean extra pairs on statute questions at most 250.
3. **Retrieval, held-out (Task 6b).** For each of the held-out formal set and the everyday-phrased set: statute questions gained minus lost at least 15% of the statute questions, with at most 2 lost; other questions at most 2 lost; statute step time p90 at most 2.5 s.
4. **Answers (Task 7).** Compare arm means on the 35 statute questions: three flag-on runs against three flag-off samples (the 2026-09-28 run plus two reruns). Flag-on minus flag-off at least +3 correct; on the 23 questions the step does not change, the difference within plus or minus 2 (that group is the noise control); on the 45 other questions the first flag-on run within 3 of the baseline 28. `final_ok` at least 75 of 80 (baseline 76). First drafts flagged by the verifier at most 13 (baseline 10). Mean turn time not above the baseline 32.7 s.
5. **Decision.** `rag.article_evidence` is switched on in `configs/local.yaml` only if 2, 3 and 4 pass. Otherwise it stays off and the results go in the doc.

## Working rules

- Work in the worktree created in Task 0, never in the main checkout: it is the owner's working copy on another branch and may hold uncommitted changes of its own.
- `PY=/home/layke/Documents/MyLLMAgent/.venv/bin/python`. Run every command from the worktree root with `PYTHONPATH=src`, otherwise the editable install imports the main checkout's code. The commands reuse the shell variables `PY` and `R` (`R=data/eval/reports/chat_pipeline`); set them again in a new shell.
- Scripts that use the live index need `LEGAL_RAG__PATHS__ROOT=/home/layke/Documents/MyLLMAgent` (or `--root`); the worktree has no `data/indexes`. Give the variable inline on the one command (`LEGAL_RAG__PATHS__ROOT=... $PY ...`) and never `export` it in a shell that also runs pytest: the environment overrides the tests' temporary paths, and `test_build_orchestrator_gives_the_step_the_retrievers_own_reranker` would write `conversations.db` into that root.
- One heavy GPU process at a time. Before Task 6 and Task 7, unload Ollama models (`curl -s localhost:11434/api/generate -d '{"model":"qwen3:4b","keep_alive":0}'`, same for `qwen3:8b`) and check `nvidia-smi` shows under 1 GB used.
- Commit after each task. End every commit message with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

---

### Task 0: Worktree, data links and baseline

**Files:**
- Create: `.worktrees/statute-article-evidence/` (git worktree, branch `feat/statute-article-evidence`)

**Step 1: Create the worktree from the current head**

The head was `claude/sft-adapter-eval` at `160b94f` when this plan was written (a later commit, `5239664`, added only docs). That branch is 23 commits ahead of `main` and holds the `/v1/chat` chain and its evaluation scripts, so open any pull request against `claude/sft-adapter-eval`, not `main`; otherwise it carries those 23 commits too.

```bash
cd /home/layke/Documents/MyLLMAgent
git worktree add .worktrees/statute-article-evidence -b feat/statute-article-evidence HEAD
cd .worktrees/statute-article-evidence
```

**Step 2: Link the untracked data the evaluation scripts read**

```bash
for f in administrative civil criminal; do ln -s /home/layke/Documents/MyLLMAgent/data/eval/$f-pilot-200.jsonl data/eval/$f-pilot-200.jsonl; done
ln -s /home/layke/Documents/MyLLMAgent/data/eval/targets data/eval/targets
ln -s /home/layke/Documents/MyLLMAgent/data/eval/reports data/eval/reports
for c in administrative civil criminal ip; do ln -s /home/layke/Documents/MyLLMAgent/data/processed/$c-v1 data/processed/$c-v1; done
git status --short
```

Expected: no output (`data/**` is ignored, so the links are not tracked).

**Step 3: Record the baseline**

```bash
PY=/home/layke/Documents/MyLLMAgent/.venv/bin/python
PYTHONPATH=src $PY -m pytest tests/legal_rag -q -p no:cacheprovider
```

Expected: `538 passed, 3 skipped`.

---

### Task 1: Split statute text into articles

**Files:**
- Modify: `src/legal_rag/ingestion/statute_articles.py`
- Test: `tests/legal_rag/unit/test_statute_articles.py`

**Step 1: Write the failing tests**

In `tests/legal_rag/unit/test_statute_articles.py` change the import on line 4 to

```python
from legal_rag.ingestion.statute_articles import ArticleSegment, StatuteArticleLocator, split_articles
```

and append at the end of the file (it reuses the file's `LAW` constant):

```python
def test_split_cuts_a_statute_at_its_headings_and_keeps_the_addenda_apart():
    segments = split_articles(LAW[LAW.index("제2조"):])
    assert [s.article for s in segments] == ["제2조", "제3조", "제4조", "제4조의2", None]
    assert segments[0].text.startswith("제2조(정의)") and "2. 제3조제2항에 따른 사람" in segments[0].text
    assert segments[-1].text.startswith("부칙") and "제1조(시행일)" in segments[-1].text
    assert all("부칙" not in s.text for s in segments[:-1])  # the addenda stay out of the last main-text article


def test_split_names_the_opening_after_the_article_the_window_opens_in():
    segments = split_articles(LAW[LAW.index("1. 부단체장"):], opens_in="제2조")
    assert [s.article for s in segments][:2] == ["제2조", "제3조"]
    assert segments[0].text.startswith("1. 부단체장")


def test_split_treats_a_cross_reference_at_a_line_start_as_body_text():
    text = "제2조(정의) 다음과 같다.\n제3조제2항에 따른 사람은 제외한다.\n제4조(권한) 본문"
    segments = split_articles(text)
    assert [s.article for s in segments] == ["제2조", "제4조"]
    assert "제3조제2항에 따른 사람은 제외한다." in segments[0].text


def test_split_counts_a_deleted_article_as_a_segment():
    text = "제4조(가) x\n제5조 삭제 <2010. 1. 1.>\n제6조(나) y"
    assert [s.article for s in split_articles(text)] == ["제4조", "제5조", "제6조"]


def test_split_returns_a_headingless_chunk_whole_and_nothing_for_blank_text():
    assert split_articles("임대차 보증금 반환 청구", opens_in="제3조의2") == [ArticleSegment("제3조의2", "임대차 보증금 반환 청구")]
    assert split_articles("   ") == []


def test_split_of_text_that_starts_at_a_heading_has_no_opening_segment():
    segments = split_articles("제1조(목적) 이 법은 목적을 정한다.\n제2조(정의) 정의한다.", opens_in="제9조")
    assert [s.article for s in segments] == ["제1조", "제2조"]
```

**Step 2: Run the tests to verify they fail**

```bash
PYTHONPATH=src $PY -m pytest tests/legal_rag/unit/test_statute_articles.py -q -p no:cacheprovider
```

Expected: collection error, `ImportError: cannot import name 'ArticleSegment'`.

**Step 3: Implement**

Apply to `src/legal_rag/ingestion/statute_articles.py`:

```diff
diff --git a/src/legal_rag/ingestion/statute_articles.py b/src/legal_rag/ingestion/statute_articles.py
index c1638fa..26db581 100644
--- a/src/legal_rag/ingestion/statute_articles.py
+++ b/src/legal_rag/ingestion/statute_articles.py
@@ -16,4 +16,5 @@ from __future__ import annotations
 
 from bisect import bisect_right
+from dataclasses import dataclass
 import re
 
@@ -55,3 +56,32 @@ class StatuteArticleLocator:
 
 
-__all__ = ["StatuteArticleLocator"]
+@dataclass(frozen=True)
+class ArticleSegment:
+    article: str | None  # "제36조"; None for addenda and for an opening whose article is unknown
+    text: str
+
+
+def split_articles(text: str, opens_in: str | None = None) -> list[ArticleSegment]:
+    """Cut statute text at its article headings, one segment per article.
+
+    Text before the first heading continues the article the window opens in (``opens_in``,
+    the ``metadata.article`` that ``StatuteArticleLocator`` filled at load time). Text from the
+    addenda on belongs to no main-text article, so its own 제1조 never names the main text.
+    Chunks with no heading, such as one-article chunks of a structured corpus, come back whole.
+    """
+    addenda = _ADDENDA.search(text)
+    end = addenda.start() if addenda else len(text)
+    headings = [(m.start(1), m.group(1)) for m in _HEADING.finditer(text) if m.start(1) < end]
+    if headings:
+        bounds = [start for start, _ in headings[1:]] + [end]
+        pieces = [(opens_in, 0, headings[0][0])] + [(article, start, stop) for (start, article), stop in zip(headings, bounds)]
+    else:
+        pieces = [(opens_in, 0, end)]
+    segments = [ArticleSegment(article, text[left:right].strip()) for article, left, right in pieces]
+    segments = [segment for segment in segments if segment.text]
+    if text[end:].strip():
+        segments.append(ArticleSegment(None, text[end:].strip()))
+    return segments
+
+
+__all__ = ["ArticleSegment", "StatuteArticleLocator", "split_articles"]
```

**Step 4: Run the tests to verify they pass**

```bash
PYTHONPATH=src $PY -m pytest tests/legal_rag/unit/test_statute_articles.py -q -p no:cacheprovider
```

Expected: `12 passed`.

**Step 5: Commit**

```bash
git add src/legal_rag/ingestion/statute_articles.py tests/legal_rag/unit/test_statute_articles.py
git commit -m "feat: split statute text at article headings" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Score arbitrary pairs with the reranker

**Files:**
- Modify: `src/legal_rag/reranking/cross_encoder.py`
- Test: `tests/legal_rag/unit/test_cross_encoder_reranker.py`

**Step 1: Write the failing tests**

Append to `tests/legal_rag/unit/test_cross_encoder_reranker.py` (it reuses the file's `_reranker` helper):

```python
def test_score_returns_one_score_per_pair_in_input_order():
    reranker, loader = _reranker({'가': 0.25, '나': 0.75})
    assert reranker.score([('q', '나'), ('q', '가'), ('q', '다')]) == [0.75, 0.25, 0.0]
    assert loader.encoder.calls[0]['batch_size'] == 16


def test_score_of_no_pairs_never_loads_the_model():
    reranker, loader = _reranker({})
    assert reranker.score([]) == []
    assert loader.calls == []
```

**Step 2: Run the tests to verify they fail**

```bash
PYTHONPATH=src $PY -m pytest tests/legal_rag/unit/test_cross_encoder_reranker.py -q -p no:cacheprovider
```

Expected: 2 failed, `AttributeError: 'CrossEncoderReranker' object has no attribute 'score'`.

**Step 3: Implement**

Apply to `src/legal_rag/reranking/cross_encoder.py` (`rerank` now calls `score`, so both share one code path):

```diff
diff --git a/src/legal_rag/reranking/cross_encoder.py b/src/legal_rag/reranking/cross_encoder.py
index e3fca7e..cf81451 100644
--- a/src/legal_rag/reranking/cross_encoder.py
+++ b/src/legal_rag/reranking/cross_encoder.py
@@ -61,9 +61,14 @@ class CrossEncoderReranker:
         return self._encoder
 
+    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
+        """Cross-encoder score of each (query, text) pair, in input order."""
+        if not pairs:
+            return []
+        return [float(score) for score in self.encoder().predict(pairs, batch_size=self.batch_size)]
+
     def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
         if k <= 0 or not candidates:
             return []
-        pairs = [(query, candidate.chunk.text) for candidate in candidates]
-        scores = [float(score) for score in self.encoder().predict(pairs, batch_size=self.batch_size)]
+        scores = self.score([(query, candidate.chunk.text) for candidate in candidates])
         order = sorted(range(len(candidates)), key=lambda index: (-scores[index], index))
         return [candidates[index].model_copy(update={'score': scores[index], 'retriever': 'reranked'})
```

**Step 4: Run the tests to verify they pass**

```bash
PYTHONPATH=src $PY -m pytest tests/legal_rag/unit/test_cross_encoder_reranker.py -q -p no:cacheprovider
```

Expected: `18 passed`.

**Step 5: Commit**

```bash
git add src/legal_rag/reranking/cross_encoder.py tests/legal_rag/unit/test_cross_encoder_reranker.py
git commit -m "feat: CrossEncoderReranker.score for arbitrary pairs" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Settings

**Files:**
- Modify: `src/legal_rag/config/settings.py`
- Test: `tests/legal_rag/unit/test_settings.py`

**Step 1: Write the failing test**

Append to `tests/legal_rag/unit/test_settings.py`:

```python
def test_article_evidence_is_off_by_default():
    rag = load_settings(profile="default").rag
    assert rag.article_evidence is False and rag.article_pool == 30 and rag.article_keep_top == 10
```

**Step 2: Run it to verify it fails**

```bash
PYTHONPATH=src $PY -m pytest tests/legal_rag/unit/test_settings.py -q -p no:cacheprovider
```

Expected: 1 failed, `AttributeError` for `article_evidence`.

**Step 3: Implement**

Apply to `src/legal_rag/config/settings.py`:

```diff
diff --git a/src/legal_rag/config/settings.py b/src/legal_rag/config/settings.py
index 5a922d1..33c17fa 100644
--- a/src/legal_rag/config/settings.py
+++ b/src/legal_rag/config/settings.py
@@ -111,2 +111,9 @@ class RagSettings(_Section):
     max_regenerations: int = 1
     max_evidence_chars: int = 1500
+    # Show the article that answers the question instead of the first max_evidence_chars of a
+    # statute window: split the windows of the top candidates at article headings, score each article
+    # with the reranker, keep the best k_final units. Needs rerank.backend: cross_encoder.
+    article_evidence: bool = False
+    # Candidates by fused rank whose statute windows are split, plus the reranker's own top hits.
+    article_pool: int = Field(default=30, ge=1)
+    article_keep_top: int = Field(default=10, ge=0)
```

**Step 4: Run the tests to verify they pass**

```bash
PYTHONPATH=src $PY -m pytest tests/legal_rag/unit/test_settings.py -q -p no:cacheprovider
```

Expected: `10 passed`.

**Step 5: Commit**

```bash
git add src/legal_rag/config/settings.py tests/legal_rag/unit/test_settings.py
git commit -m "feat: rag.article_evidence settings, off by default" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The article evidence selector

**Files:**
- Create: `src/legal_rag/rag/article_evidence.py`
- Create: `tests/legal_rag/unit/test_article_evidence.py`

**Step 1: Write the failing tests**

Create `tests/legal_rag/unit/test_article_evidence.py` (it reuses the fake encoder and loader from `test_cross_encoder_reranker.py`):

```python
"""Article-level evidence selection with a fake cross-encoder: no model, no network."""
from __future__ import annotations

from legal_rag.ingestion.statute_articles import split_articles
from legal_rag.rag.article_evidence import ArticleEvidenceSelector
from legal_rag.reranking.cross_encoder import CrossEncoderReranker
from legal_rag.schemas import Chunk, ChunkMetadata, DocumentType, ScoredChunk
from tests.legal_rag.unit.test_cross_encoder_reranker import _FakeEncoder, _FakeLoader

WINDOW = (
    "제35조(사업의 승계) 사업자가 죽으면 상속인이 승계한다.\n"
    "제36조(사업의 휴업ㆍ폐업)\n① 사업자가 휴업하려면 허가를 받아야 한다.\n⑤ 휴업하려는 경우 인터넷 홈페이지에 게시하여야 한다.\n"
    "제37조(면허취소) 면허를 취소할 수 있다.\n"
)
QUERY = "휴업 또는 폐업하려는 경우 어떻게 공지해야 하나요?"


def _chunk(chunk_id, text, document_id="law", document_type=DocumentType.statute, article=None, paragraph=None):
    statute = document_type == DocumentType.statute
    return Chunk(id=chunk_id, document_id=document_id, document_type=document_type, path="chunk:0000", order=0, text=text,
                 title="도시철도법", metadata=ChunkMetadata(dataset="d", corpus_version="v", law_name="도시철도법" if statute else None,
                                                        article=article, paragraph=paragraph))


def _hit(chunk, score):
    return ScoredChunk(chunk=chunk, score=score, retriever="reranked")


def _scores(text, opens_in=None, **by_label):
    """A score for every article of ``text``: by_label maps '36' to the score of 제36조, others get 0.1."""
    return {s.text: by_label.get((s.article or "")[1:-1], 0.1) for s in split_articles(text, opens_in)}


def _selector(scores, **kwargs):
    loader = _FakeLoader(_FakeEncoder(scores))
    reranker = CrossEncoderReranker(model="fake", device="cpu", loader=loader)
    return ArticleEvidenceSelector(reranker, **kwargs), loader


def test_a_statute_window_becomes_one_unit_per_article_best_first():
    window = _chunk("w1", WINDOW)
    selector, _ = _selector(_scores(WINDOW, **{"36": 0.9}))
    units = selector.select(QUERY, [_hit(window, 0.05)], ["w1"], k=2)
    assert [u.chunk.metadata.article for u in units][0] == "제36조" and len(units) == 2
    best = units[0]
    assert best.chunk.text.startswith("제36조(사업의 휴업ㆍ폐업)") and "인터넷 홈페이지에 게시" in best.chunk.text
    assert best.chunk.id == "w1@1" and best.score == 0.9 and best.retriever == "article"
    assert best.chunk.document_id == "law" and best.chunk.metadata.law_name == "도시철도법"
    assert window.text == WINDOW  # the retrieved chunk is untouched
    assert selector.select(QUERY, [_hit(window, 0.05)], ["w1"], k=0) == []  # like the reranker, k <= 0 selects nothing


def test_other_chunks_stay_whole_and_compete_on_their_reranker_score():
    case = _chunk("c1", "대법원 판결 이유", document_id="case", document_type=DocumentType.court_decision)
    window = _chunk("w1", WINDOW)
    selector, _ = _selector(_scores(WINDOW, **{"36": 0.9}))
    hits = [_hit(case, 0.8), _hit(window, 0.05)]
    units = selector.select(QUERY, hits, ["c1", "w1"], k=2)
    assert units[0].chunk.metadata.article == "제36조" and units[1] is hits[0]


def test_without_a_statute_window_the_reranker_top_k_is_returned_untouched():
    hits = [_hit(_chunk(f"c{i}", f"판결 {i}", document_id=f"d{i}", document_type=DocumentType.court_decision), 1 - i / 10) for i in range(4)]
    selector, loader = _selector({})
    assert selector.select(QUERY, hits, [h.chunk.id for h in hits], k=3) == hits[:3]
    assert loader.encoder.calls == []


def test_candidates_are_the_fused_top_plus_the_rerankers_top():
    windows = [_chunk(f"w{i}", f"제{i}조(가) 본문 {i}", document_id=f"d{i}") for i in range(4)]
    hits = [_hit(w, 1 - i / 10) for i, w in enumerate(windows)]  # reranker order w0 w1 w2 w3
    selector, loader = _selector({}, pool=1, keep_top=1)
    units = selector.select(QUERY, hits, ["w3", "w2", "w1", "w0"], k=4)  # fused order reversed
    pairs = loader.encoder.calls[0]["pairs"]
    assert {query for query, _ in pairs} == {QUERY}  # articles are scored against the question
    assert sorted(text for _, text in pairs) == sorted([windows[0].text, windows[3].text])
    assert {u.chunk.id.split("@")[0] for u in units} == {"w0", "w3"}  # nothing outside the pool comes back


def test_an_article_held_by_two_overlapping_windows_is_kept_once_at_its_best_score():
    first = _chunk("w1", WINDOW)
    second_text = WINDOW.replace("게시하여야 한다.", "게시하여야 한다. 다만 천재지변은 제외한다.") + "제38조(벌칙) 벌금에 처한다.\n"
    second = _chunk("w2", second_text)
    scores = {**_scores(WINDOW, **{"36": 0.7}), **_scores(second_text, **{"36": 0.9})}
    selector, _ = _selector(scores)
    other_law = _chunk("w3", WINDOW, document_id="other")  # its 제36조 is another article
    units = selector.select(QUERY, [_hit(first, 0.5), _hit(second, 0.4), _hit(other_law, 0.3)], ["w1", "w2", "w3"], k=10)
    labelled = [(u.chunk.document_id, u.score) for u in units if u.chunk.metadata.article == "제36조"]
    assert labelled == [("law", 0.9), ("other", 0.7)]


def test_the_opening_of_a_window_is_named_after_the_article_it_continues():
    text = "② 계속되는 조문 본문\n" + WINDOW
    window = _chunk("w1", text, article="제34조")
    selector, _ = _selector(_scores(text, "제34조", **{"34": 0.95}))
    units = selector.select(QUERY, [_hit(window, 0.01)], ["w1"], k=1)
    assert units[0].chunk.metadata.article == "제34조" and units[0].chunk.text.startswith("② 계속되는 조문 본문")


def test_ties_keep_the_rerankers_order():
    cases = [_chunk(f"c{i}", f"판결 {i}", document_id=f"d{i}", document_type=DocumentType.court_decision) for i in range(3)]
    window = _chunk("w1", "제1조(가) 본문")
    selector, _ = _selector({"제1조(가) 본문": 0.5})
    hits = [_hit(cases[0], 0.5), _hit(cases[1], 0.5), _hit(window, 0.1), _hit(cases[2], 0.5)]
    units = selector.select(QUERY, hits, [h.chunk.id for h in reversed(hits)], k=4)  # fused order is the reverse
    assert [u.chunk.id for u in units] == ["c0", "c1", "w1@0", "c2"]


def test_paragraph_chunks_of_one_article_are_kept_apart():
    first, second = (_chunk(f"p{n}", f"{n}항 본문", article="제36조", paragraph=f"제{n}항") for n in (1, 2))
    selector, _ = _selector({})
    units = selector.select(QUERY, [_hit(first, 0.5), _hit(second, 0.4)], ["p1", "p2"], k=5)
    assert [u.chunk.id for u in units] == ["p1@0", "p2@0"]
```

**Step 2: Run the tests to verify they fail**

```bash
PYTHONPATH=src $PY -m pytest tests/legal_rag/unit/test_article_evidence.py -q -p no:cacheprovider
```

Expected: collection error, `ModuleNotFoundError: No module named 'legal_rag.rag.article_evidence'`.

**Step 3: Implement**

Create `src/legal_rag/rag/article_evidence.py`:

```python
"""Evidence at article level: show the article that answers the question, not a statute window's opening.

The imported corpora cut a statute into windows of about 3,500 characters that hold several articles, and the
context builder shows the first ``max_evidence_chars`` of each. The cross-encoder reads only the first ~900
characters of a window, so a window whose answer sits further in scores low, and when it does reach the top five
the answer can still lie past the cut. Splitting the windows at article headings and scoring every article
against the question puts the cited article first (docs/superpowers/plans/2026-09-29-statute-article-evidence.md).
"""

from __future__ import annotations

from collections.abc import Sequence

from legal_rag.ingestion.statute_articles import ArticleSegment, split_articles
from legal_rag.reranking.cross_encoder import CrossEncoderReranker
from legal_rag.schemas import CitationKind, ScoredChunk, citation_kind


class ArticleEvidenceSelector:
    def __init__(self, reranker: CrossEncoderReranker, *, pool: int = 30, keep_top: int = 10) -> None:
        self.reranker = reranker
        self.pool = pool
        self.keep_top = keep_top

    def select(self, query: str, hits: Sequence[ScoredChunk], fused_ids: Sequence[str], k: int) -> list[ScoredChunk]:
        """The best ``k`` evidence units among ``hits``, the reranker's candidates in its order.

        Candidates are the first ``pool`` chunks by fused rank plus the reranker's own top ``keep_top``. A statute
        window becomes one unit per article, scored by the cross-encoder against ``query``; every other chunk stays
        whole with its reranker score, so ``query`` must be the text the reranker scored ``hits`` against. An
        article that two overlapping windows both contain is kept once, at its best score; chunks that carry
        different paragraphs of one article are kept apart. With no statute window among the candidates the
        result is ``hits[:k]``, and with ``k <= 0`` it is empty.
        """
        if k <= 0:
            return []
        in_pool = set(fused_ids[: self.pool])
        candidates = [hit for rank, hit in enumerate(hits) if rank < self.keep_top or hit.chunk.id in in_pool]
        statute = [hit for hit in candidates if citation_kind(hit.chunk.document_type) is CitationKind.statute]
        if not statute:
            return list(hits[:k])
        split = {hit.chunk.id: split_articles(hit.chunk.text, hit.chunk.metadata.article) for hit in statute}
        pieces = [(hit, n, segment) for hit in statute for n, segment in enumerate(split[hit.chunk.id])]
        scores = self.reranker.score([(query, segment.text) for _, _, segment in pieces])
        articles: dict[str, list[ScoredChunk]] = {hit.chunk.id: [] for hit in statute}
        for (hit, n, segment), score in zip(pieces, scores):
            articles[hit.chunk.id].append(_unit(hit, n, segment, score))
        units = [unit for hit in candidates for unit in (articles.get(hit.chunk.id) or [hit])]  # reranker order kept
        chosen: list[ScoredChunk] = []
        seen: set[tuple[str, str, str | None]] = set()
        for unit in sorted(units, key=lambda unit: -unit.score):  # stable: ties keep the reranker's order
            if unit.retriever == "article" and unit.chunk.metadata.article:
                article = (unit.chunk.document_id, unit.chunk.metadata.article, unit.chunk.metadata.paragraph)
                if article in seen:
                    continue
                seen.add(article)
            chosen.append(unit)
            if len(chosen) == k:
                break
        return chosen


def _unit(hit: ScoredChunk, n: int, segment: ArticleSegment, score: float) -> ScoredChunk:
    chunk = hit.chunk.model_copy(update={
        "id": f"{hit.chunk.id}@{n}",  # audit only; nothing looks a chunk up by id
        "text": segment.text,
        "metadata": hit.chunk.metadata.model_copy(update={"article": segment.article}),
    })
    return ScoredChunk(chunk=chunk, score=score, retriever="article")
```

**Step 4: Run the tests to verify they pass**

```bash
PYTHONPATH=src $PY -m pytest tests/legal_rag/unit/test_article_evidence.py -q -p no:cacheprovider
```

Expected: `8 passed`.

**Step 5: Commit**

```bash
git add src/legal_rag/rag/article_evidence.py tests/legal_rag/unit/test_article_evidence.py
git commit -m "feat: ArticleEvidenceSelector scores statute articles instead of window heads" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Wire the step into the orchestrator and the factory

**Files:**
- Modify: `src/legal_rag/rag/orchestrator.py`
- Modify: `src/legal_rag/factory.py`
- Modify: `src/legal_rag/retrieval/hybrid.py` (the retriever's trace becomes per thread; see Step 3)
- Test: `tests/legal_rag/unit/test_article_evidence.py`, `tests/legal_rag/integration/test_chat_turn.py`, `tests/legal_rag/unit/test_fusion.py`

**Step 1: Write the failing tests**

In `tests/legal_rag/unit/test_article_evidence.py` add these imports at the top (below `from __future__ import annotations`), keeping the others:

```python
import os

import pytest

from legal_rag.config import load_settings
from legal_rag.factory import build_evidence_selector
from legal_rag.reranking import NoopReranker
```

and append at the end:

```python
@pytest.fixture(autouse=True)
def _no_legal_rag_env(monkeypatch):
    """Settings put the environment above overrides: a LEGAL_RAG__* variable exported in the caller's shell must not leak in."""
    for name in [name for name in os.environ if name.startswith("LEGAL_RAG__")]:
        monkeypatch.delenv(name)


class _Retriever:
    """Only the reranker matters to the factory."""

    def __init__(self, reranker):
        self.reranker = reranker


def test_factory_builds_the_step_only_when_enabled_and_only_over_a_cross_encoder():
    assert build_evidence_selector(load_settings(profile="default"), _Retriever(NoopReranker())) is None
    enabled = load_settings(profile="default", overrides={"rag": {"article_evidence": True, "article_pool": 7, "article_keep_top": 4}})
    with pytest.raises(ValueError, match="cross_encoder"):
        build_evidence_selector(enabled, _Retriever(NoopReranker()))
    reranker = CrossEncoderReranker(model="fake", device="cpu", loader=_FakeLoader(_FakeEncoder({})))
    selector = build_evidence_selector(enabled, _Retriever(reranker))
    assert selector.reranker is reranker and (selector.pool, selector.keep_top) == (7, 4)


def test_build_orchestrator_gives_the_step_the_retrievers_own_reranker(tmp_path):
    from legal_rag.factory import build_orchestrator
    from legal_rag.retrieval.fusion import RRFFuser
    from legal_rag.retrieval.hybrid import HybridRetriever

    overrides = {"paths": {"root": str(tmp_path)}, "llm": {"backend": "fake"}, "rag": {"article_evidence": True}}
    reranker = CrossEncoderReranker(model="fake", device="cpu", loader=_FakeLoader(_FakeEncoder({})))
    retriever = HybridRetriever([], RRFFuser(60), reranker, k_leg=1, k_fused=1)
    orchestrator = build_orchestrator(load_settings(profile="default", overrides=overrides), retriever=retriever)
    assert orchestrator.retriever is retriever and orchestrator.evidence_selector.reranker is reranker
    off = build_orchestrator(load_settings(profile="default", overrides={**overrides, "rag": {"article_evidence": False}}), retriever=retriever)
    assert off.retriever is retriever and off.evidence_selector is None
```

In `tests/legal_rag/integration/test_chat_turn.py` add `from legal_rag.retrieval.hybrid import HybridRetriever` under the `Orchestrator` import and append (it reuses the file's `retriever` fixture, `_answer` and `_orchestrator`):

```python
class _ArticleStep:
    """Stands in for ArticleEvidenceSelector: records what the orchestrator hands it, keeps one unit."""

    def __init__(self):
        self.calls = []

    def select(self, query, hits, fused_ids, k):
        self.calls.append((query, [h.chunk.id for h in hits], list(fused_ids), k))
        return list(hits)[:1]


def test_article_step_gets_every_reranked_candidate_and_replaces_the_evidence(retriever):
    settings, r = retriever
    step = _ArticleStep()
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis(domain="민사", issues=["보증금 반환"], keywords=["임차권등기명령"])], "LegalAnswer": [_answer()]})
    orch = Orchestrator.from_components(settings=settings, retriever=r, model=model, store=InMemoryConversationStore(), verifier=Verifier(),
                                        policy=VerificationPolicy(max_regenerations=settings.rag.max_regenerations), evidence_selector=step)
    resp = orch.handle("임차권등기명령 신청서에 적어야 하는 사항은?", conversation_id="c-art")
    [(query, hit_ids, fused_ids, k)] = step.calls
    assert query == "임차권등기명령 신청서에 적어야 하는 사항은?" and k == settings.retrieval.k_final
    assert len(hit_ids) > k and set(hit_ids) == set(fused_ids)  # depth is k_fused, not k_final
    assert len(resp.evidence.items) == 1 and resp.evidence.items[0].chunk_id == hit_ids[0]
    assert any(timing.stage == "select_articles" for timing in resp.trace.timings)


def test_without_the_article_step_retrieval_stops_at_k_final(retriever):
    settings, r = retriever
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis(domain="민사", issues=["보증금 반환"], keywords=["임차권등기명령"])], "LegalAnswer": [_answer()]})
    resp = _orchestrator(settings, r, model).handle("임차권등기명령 신청서에 적어야 하는 사항은?", conversation_id="c-noart")
    assert len(resp.evidence.items) == settings.retrieval.k_final
    assert not any(timing.stage == "select_articles" for timing in resp.trace.timings)


class _ReverseReranker:
    """Reorders the fused candidates, so that the reranked order differs from the fused order."""

    name = "reverse"

    def rerank(self, query, candidates, k):
        return [candidate.model_copy(update={"retriever": "reranked"}) for candidate in reversed(candidates)][:k]


def test_article_step_pool_follows_the_fused_ranking_not_the_rerankers(retriever):
    settings, r = retriever
    reordering = HybridRetriever(r.legs, r.fuser, _ReverseReranker(), k_leg=r.k_leg, k_fused=r.k_fused)
    step = _ArticleStep()
    # The reversed ranking may put an unrelated unit first, so the draft can fail verification and be regenerated.
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis(domain="민사", issues=["보증금 반환"], keywords=["임차권등기명령"])], "LegalAnswer": [_answer(), _answer()]})
    orch = Orchestrator.from_components(settings=settings, retriever=reordering, model=model, store=InMemoryConversationStore(), verifier=Verifier(),
                                        policy=VerificationPolicy(max_regenerations=settings.rag.max_regenerations), evidence_selector=step)
    resp = orch.handle("임차권등기명령 신청서에 적어야 하는 사항은?", conversation_id="c-fused")
    [(_, hit_ids, fused_ids, _)] = step.calls
    assert fused_ids == [chunk_id for chunk_id, _ in resp.trace.retrieved["fused"]]
    assert hit_ids == list(reversed(fused_ids))  # the reranker's order is not the fused order


def test_article_step_is_skipped_for_a_retriever_that_is_not_hybrid(retriever):
    settings, r = retriever

    class _Flat:
        name = "flat"

        def retrieve(self, query, k, filters=None):
            return r.retrieve(query, k, filters)

    step = _ArticleStep()
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis(domain="민사", issues=["보증금 반환"], keywords=["임차권등기명령"])], "LegalAnswer": [_answer()]})
    orch = Orchestrator.from_components(settings=settings, retriever=_Flat(), model=model, store=InMemoryConversationStore(), verifier=Verifier(),
                                        policy=VerificationPolicy(max_regenerations=settings.rag.max_regenerations), evidence_selector=step)
    resp = orch.handle("임차권등기명령 신청서에 적어야 하는 사항은?", conversation_id="c-flat")
    assert step.calls == [] and len(resp.evidence.items) == settings.retrieval.k_final
```

The orchestrator reads the fused ranking from `HybridRetriever.last_trace` right after retrieval. `/v1/chat` runs `handle` on a thread pool over one shared retriever, and `retrieve_many` resets that attribute at the start of every call, so an overlapping request would replace the dict while this request's reranker is still running. Today that only garbles the audit trace; with the article step it would silently shrink the pool to the reranker's top 10 for the affected request. The trace therefore has to belong to the calling thread. In `tests/legal_rag/unit/test_fusion.py` add `import threading` above the first import and append at the end:

```python
class _ByQuery:
    name = "dense"

    def __init__(self, hits_by_query):
        self.hits_by_query = hits_by_query

    def retrieve(self, query, k, filters=None):
        return self.hits_by_query[query][:k]


def test_last_trace_belongs_to_the_thread_that_ran_the_retrieval():
    """/v1/chat runs handle() on a thread pool over one shared retriever: a request must read its own fused list."""
    c1, c2, _ = _chunks()
    leg = _ByQuery({"first": [_sc(c1, 0.9, "dense")], "second": [_sc(c2, 0.9, "dense")]})
    inside, resume = threading.Event(), threading.Event()

    class _Pausing:
        def rerank(self, query, candidates, k):
            if query == "first":
                inside.set()  # the first request is between fusing and reranking
                assert resume.wait(5)
            return list(candidates)[:k]

    hybrid = HybridRetriever([leg], RRFFuser(60), _Pausing(), k_leg=5, k_fused=5)
    seen = {}

    def first_request():
        hybrid.retrieve_many(["first"], k=1)
        seen["fused"] = [chunk_id for chunk_id, _ in hybrid.last_trace["fused"]]

    worker = threading.Thread(target=first_request)
    worker.start()
    assert inside.wait(5)
    hybrid.retrieve_many(["second"], k=1)  # a second request runs to the end in the meantime
    resume.set()
    worker.join(5)
    assert seen["fused"] == [c1.id]
    assert [chunk_id for chunk_id, _ in hybrid.last_trace["fused"]] == [c2.id]
```

**Step 2: Run the tests to verify they fail**

```bash
for f in unit/test_article_evidence.py integration/test_chat_turn.py unit/test_fusion.py; do
  PYTHONPATH=src $PY -m pytest tests/legal_rag/$f -q -p no:cacheprovider 2>&1 | tail -4
done
```

One run per file, because a collection error in one file stops pytest before it runs the others. Expected: the unit file fails to import `build_evidence_selector` (`ImportError`, 1 error); the chat-turn file has 3 failed, each `TypeError: from_components() got an unexpected keyword argument 'evidence_selector'` (`test_without_the_article_step_retrieval_stops_at_k_final` passes already); the fusion file has 1 failed, `AssertionError: assert ['d2#p'] == ['d1#p']` (the first request read the second request's fused list).

**Step 3: Implement**

Apply to `src/legal_rag/rag/orchestrator.py` (an optional selector; with none, behaviour is byte-for-byte today's):

```diff
diff --git a/src/legal_rag/rag/orchestrator.py b/src/legal_rag/rag/orchestrator.py
index 72d1181..0381baf 100644
--- a/src/legal_rag/rag/orchestrator.py
+++ b/src/legal_rag/rag/orchestrator.py
@@ -4,4 +4,5 @@ from __future__ import annotations
 
 from datetime import datetime, timezone
+from typing import TYPE_CHECKING
 
 from legal_rag.config import Settings
@@ -19,4 +20,7 @@ from .rewriter import QueryRewriter
 from .trace import TraceRecorder
 
+if TYPE_CHECKING:
+    from .article_evidence import ArticleEvidenceSelector
+
 
 def _now() -> datetime:
@@ -38,4 +42,5 @@ class Orchestrator:
         store: ConversationStore,
         settings: Settings,
+        evidence_selector: "ArticleEvidenceSelector | None" = None,
     ) -> None:
         self.analyzer = analyzer
@@ -49,7 +54,8 @@ class Orchestrator:
         self.store = store
         self.settings = settings
+        self.evidence_selector = evidence_selector
 
     @classmethod
-    def from_components(cls, *, settings: Settings, retriever: Retriever, model: ChatModel, store: ConversationStore, verifier: Verifier, policy: VerificationPolicy) -> "Orchestrator":
+    def from_components(cls, *, settings: Settings, retriever: Retriever, model: ChatModel, store: ConversationStore, verifier: Verifier, policy: VerificationPolicy, evidence_selector: "ArticleEvidenceSelector | None" = None) -> "Orchestrator":
         return cls(
             analyzer=QueryAnalyzer(model, history_turns=settings.rag.history_turns),
@@ -64,4 +70,5 @@ class Orchestrator:
             store=store,
             settings=settings,
+            evidence_selector=evidence_selector,
         )
 
@@ -93,10 +100,16 @@ class Orchestrator:
 
         k = self.settings.retrieval.k_final
+        articles = self.evidence_selector is not None and isinstance(self.retriever, HybridRetriever)
         with recorder.stage("retrieve"):
             if isinstance(self.retriever, HybridRetriever):
-                hits = self.retriever.retrieve_many(queries.search_queries, k, filters, rerank_query=message)
+                # The article step needs every candidate the reranker scored, not only its top k.
+                depth = self.settings.retrieval.k_fused if articles else k
+                hits = self.retriever.retrieve_many(queries.search_queries, depth, filters, rerank_query=message)
             else:
                 hits = self.retriever.retrieve(message, k, filters)
         trace.retrieved = dict(getattr(self.retriever, "last_trace", {}) or {})
+        if articles:
+            with recorder.stage("select_articles"):
+                hits = self.evidence_selector.select(message, hits, [cid for cid, _ in trace.retrieved.get("fused", [])], k)
 
         with recorder.stage("context"):
```

Apply to `src/legal_rag/factory.py` (the selector shares the retriever's reranker, so one copy of the model stays on the GPU):

```diff
diff --git a/src/legal_rag/factory.py b/src/legal_rag/factory.py
index 246409e..b0c14b6 100644
--- a/src/legal_rag/factory.py
+++ b/src/legal_rag/factory.py
@@ -208,4 +208,17 @@ def build_chat_model(settings: Settings):
 
 
+def build_evidence_selector(settings: Settings, retriever):
+    """The article-level evidence step sharing the retriever's reranker, or None when it is off."""
+    if not settings.rag.article_evidence:
+        return None
+    from legal_rag.rag.article_evidence import ArticleEvidenceSelector
+    from legal_rag.reranking.cross_encoder import CrossEncoderReranker
+
+    reranker = getattr(retriever, "reranker", None)
+    if not isinstance(reranker, CrossEncoderReranker):
+        raise ValueError("rag.article_evidence needs rerank.backend: cross_encoder")
+    return ArticleEvidenceSelector(reranker, pool=settings.rag.article_pool, keep_top=settings.rag.article_keep_top)
+
+
 def build_orchestrator(settings: Settings, *, retriever: HybridRetriever | None = None):
     from legal_rag.rag.conversation import SQLiteConversationStore
@@ -214,7 +227,9 @@ def build_orchestrator(settings: Settings, *, retriever: HybridRetriever | None
 
     settings.index_corpus_dir().mkdir(parents=True, exist_ok=True)
+    retriever = retriever or build_hybrid_retriever(settings)
     return Orchestrator.from_components(
         settings=settings,
-        retriever=retriever or build_hybrid_retriever(settings),
+        retriever=retriever,
+        evidence_selector=build_evidence_selector(settings, retriever),
         model=build_chat_model(settings),
         store=SQLiteConversationStore(settings.index_corpus_dir() / "conversations.db"),
```

Apply to `src/legal_rag/retrieval/hybrid.py` (the trace of `retrieve_many` is kept per thread; a caller that reads `last_trace` in the thread that ran the retrieval sees the same dict as before):

```diff
diff --git a/src/legal_rag/retrieval/hybrid.py b/src/legal_rag/retrieval/hybrid.py
index 59b35aa..d6ead14 100644
--- a/src/legal_rag/retrieval/hybrid.py
+++ b/src/legal_rag/retrieval/hybrid.py
@@ -2,6 +2,7 @@
 
 from __future__ import annotations
 
+import threading
 from typing import Protocol
 
 from legal_rag.reranking import Reranker
@@ -25,7 +26,21 @@ class HybridRetriever:
         self.reranker = reranker
         self.k_leg = k_leg
         self.k_fused = k_fused
-        self.last_trace: dict[str, list[tuple[str, float]]] = {}
+        self._trace = threading.local()
+        self.last_trace = {}
+
+    @property
+    def last_trace(self) -> dict[str, list[tuple[str, float]]]:
+        """The lists of this thread's latest retrieve_many: one per leg and query, plus "fused" and "reranked".
+
+        Per thread because /v1/chat runs the orchestrator on a thread pool over one shared retriever, and a
+        single shared dict would hand a request the fused list of another request's query.
+        """
+        return getattr(self._trace, "lists", {})
+
+    @last_trace.setter
+    def last_trace(self, lists: dict[str, list[tuple[str, float]]]) -> None:
+        self._trace.lists = lists
 
     def retrieve(self, query: str, k: int, filters: RetrievalFilters | None = None) -> list[ScoredChunk]:
         return self.retrieve_many([query], k, filters)
```

**Step 4: Run the tests, then the whole backend suite**

```bash
PYTHONPATH=src $PY -m pytest tests/legal_rag/unit/test_article_evidence.py tests/legal_rag/integration/test_chat_turn.py tests/legal_rag/unit/test_fusion.py -q -p no:cacheprovider
PYTHONPATH=src $PY -m pytest tests/legal_rag -q -p no:cacheprovider
```

Expected: `36 passed` for the first command; `562 passed, 3 skipped` for the second.

**Step 5: Commit**

```bash
git add src/legal_rag/rag/orchestrator.py src/legal_rag/factory.py src/legal_rag/retrieval/hybrid.py tests/legal_rag/unit/test_article_evidence.py tests/legal_rag/integration/test_chat_turn.py tests/legal_rag/unit/test_fusion.py
git commit -m "feat: run the article evidence step between retrieval and the context builder" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Retrieval-only acceptance script

**Files:**
- Create: `scripts/eval_article_evidence.py`

The script replays the analyzer, rule rewriter and retrieval as `handle()` does, builds today's evidence and the article step's from the same candidates, and reports coverage, step time and pairs. It needs no generation.

**Step 1: Create the script**

```python
#!/usr/bin/env python
"""Does article-level evidence show the article the reference cites? Retrieval only, no generation.

Reads a saved chat-pipeline run (answer_pairs.jsonl: question, AIHub reference answer, gold document). For every question it
replays the analyzer, the rule rewriter and retrieval as handle() does, then builds the evidence twice from the same candidates:
today's (the reranker's top k_final chunks) and the article step's. A statute question is covered when a shown item of the gold
statute carries an article the reference cites (header Article, or an article heading inside the shown text); any other question
is covered when a shown item comes from the gold document. Also times the article step and counts its cross-encoder pairs.

Example (one heavy process at a time; unload Ollama models first):
    uv run python scripts/eval_article_evidence.py --run data/eval/reports/chat_pipeline/formal-2026-09-28-articles \
        --out data/eval/reports/chat_pipeline/article-coverage
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import time
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

app = typer.Typer(add_completion=False)
CITED = re.compile(r"제\s*(\d+)\s*조(?:\s*의\s*(\d+))?")


def cited_articles(reference: str) -> set[str]:
    return {f"제{number}조" + (f"의{branch}" if branch else "") for number, branch in CITED.findall(reference)}


def shown_articles(bundle, gold: str) -> set[str]:
    from legal_rag.ingestion.statute_articles import split_articles

    found: set[str] = set()
    for item in bundle.items:
        if item.document_id == gold:
            found |= {segment.article for segment in split_articles(item.text, item.header.get("Article")) if segment.article}
    return found


@app.command()
def main(
    run: Path = typer.Option(..., exists=True, file_okay=False, help="A chat-pipeline run directory holding answer_pairs.jsonl."),
    out: Path = typer.Option(..., help="Directory for article_coverage.jsonl."),
    root: Path = typer.Option(None, help="Repository root that holds data/ when the code runs from another checkout."),
    limit: int = typer.Option(0, help="Only the first N questions (smoke test)."),
    pool: int = typer.Option(None, help="Override rag.article_pool."),
) -> None:
    from legal_rag.cli import settings_from_options
    from legal_rag.factory import build_orchestrator

    extra = {"rag": {"allow_clarification": False, "article_evidence": True}}
    if pool is not None:
        extra["rag"]["article_pool"] = pool
    orch = build_orchestrator(settings_from_options(profile="local", root=root, extra=extra))
    pairs = [json.loads(line) for line in open(run / "answer_pairs.jsonl", encoding="utf-8") if line.strip()]
    pairs = pairs[:limit] if limit else pairs
    counter = {"pairs": 0}
    score = orch.evidence_selector.reranker.score

    def counting(items):
        counter["pairs"] += len(items)
        return score(items)

    orch.evidence_selector.reranker.score = counting
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with (out / "article_coverage.jsonl").open("w", encoding="utf-8") as handle:
        for pair in pairs:
            question, gold = pair["question"], pair["gold"][0]
            statute = "::statute::" in gold
            analysis = orch.analyzer.analyze(question, [])
            queries = orch.rewriter.rewrite(question, analysis)
            k = orch.settings.retrieval.k_final
            hits = orch.retriever.retrieve_many(queries.search_queries, orch.settings.retrieval.k_fused, None, rerank_query=question)
            fused = [chunk_id for chunk_id, _ in orch.retriever.last_trace["fused"]]
            today = orch.context_builder.build(hits[:k])
            counter["pairs"] = 0
            started = time.perf_counter()
            units = orch.evidence_selector.select(question, hits, fused, k)
            seconds = time.perf_counter() - started
            new = orch.context_builder.build(units)
            cited = cited_articles(pair["reference"])
            if statute:
                before, after = bool(cited & shown_articles(today, gold)), bool(cited & shown_articles(new, gold))
            else:
                before, after = any(i.document_id == gold for i in today.items), any(i.document_id == gold for i in new.items)
            row = {"key": pair["key"], "statute": statute, "covered_today": before, "covered_articles": after,
                   "select_seconds": round(seconds, 3), "select_pairs": counter["pairs"],
                   "chars_today": sum(len(i.text) for i in today.items), "chars_articles": sum(len(i.text) for i in new.items)}
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
    for label, selected in (("statute questions", [r for r in rows if r["statute"]]), ("other questions", [r for r in rows if not r["statute"]])):
        if not selected:
            continue
        gained = sum(r["covered_articles"] and not r["covered_today"] for r in selected)
        lost = sum(r["covered_today"] and not r["covered_articles"] for r in selected)
        print(f"{label}: n={len(selected)}  covered today {sum(r['covered_today'] for r in selected)}  with article step "
              f"{sum(r['covered_articles'] for r in selected)}  (+{gained} / -{lost})  extra cross-encoder pairs "
              f"{statistics.mean(r['select_pairs'] for r in selected):.0f}  step time {statistics.mean(r['select_seconds'] for r in selected):.2f}s  "
              f"evidence chars {statistics.mean(r['chars_today'] for r in selected):.0f} -> {statistics.mean(r['chars_articles'] for r in selected):.0f}")


if __name__ == "__main__":
    app()
```

**Step 2: Smoke test on three questions**

```bash
PYTHONPATH=src $PY scripts/eval_article_evidence.py --run data/eval/reports/chat_pipeline/formal-2026-09-28-articles --out data/eval/reports/chat_pipeline/article-coverage-smoke --root /home/layke/Documents/MyLLMAgent --limit 3
```

Expected: two summary lines, one for statute questions (n=1) and one for the others (n=2); no traceback.

**Step 3: Run all 80 questions (about 10 minutes)**

```bash
PYTHONPATH=src $PY scripts/eval_article_evidence.py --run data/eval/reports/chat_pipeline/formal-2026-09-28-articles --out data/eval/reports/chat_pipeline/article-coverage --root /home/layke/Documents/MyLLMAgent
```

Expected, verified on 2026-09-29 (the analyzer may differ by a token between runs, so accept plus or minus one question):

```
statute questions: n=35  covered today 16  with article step 28  (+12 / -0)  extra cross-encoder pairs 169  step time 1.27s  evidence chars 7342 -> 3537
other questions: n=45  covered today 23  with article step 23  (+0 / -0)  extra cross-encoder pairs 20  step time 0.21s  evidence chars 6545 -> 6479
```

The branch's own run on 2026-09-30 (code of Tasks 1 to 5 as committed, after the review fixes) gave the same coverage numbers (statute 16 to 28, +12 / -0; other 23 to 23), 169 and 20 extra pairs, evidence characters 7342 to 3537 and 6545 to 6479, and step times of 1.44 s (p90 2.26 s) for statute questions and 0.24 s for the others. Step times move with the state of the GPU; criterion 2 bounds the mean.

Check the output against acceptance criterion 2. If it fails, stop and report; do not go on to Task 6b or Task 7.

The run writes one row per question. Keep the statute questions whose cited article the step newly shows; Task 7 needs them:

```bash
python3 - <<'EOF'
import json
folder = "data/eval/reports/chat_pipeline/article-coverage/"
rows = [json.loads(line) for line in open(folder + "article_coverage.jsonl", encoding="utf-8")]
gained = [row["key"] for row in rows if row["statute"] and row["covered_articles"] and not row["covered_today"]]
json.dump(gained, open(folder + "gained.json", "w"))
print(len(gained), "gained questions")
EOF
```

Expected: `12 gained questions`.

**Step 4: Commit**

```bash
git add scripts/eval_article_evidence.py
git commit -m "eval: retrieval-only coverage of the cited statute article, with and without the article step" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6b: Held-out retrieval check

Task 6 uses the 80 questions the design was tuned on, and they are formal AIHub questions written from the gold document. This task runs the same step, pool unchanged, on questions that had no part in the design: the 169 formal questions of the 210-question sample that are not among the 80, and the 150 everyday-phrased questions, where the retrieval query is the user's wording and the reference answers the formal question it was rewritten from.

**Files:**
- Create: `scripts/build_article_heldout.py`

**Step 1: Create the script**

```python
#!/usr/bin/env python
"""Question sets for the retrieval-only check on data the article step's pool size was not tuned on.

`formal`: the 210-question formal sample of eval_chat_pipeline (50 per domain plus crossdomain-60) minus the 80 questions of the answer
run, so none of them shaped the design. `everyday`: everyday-150, whose retrieval query is the everyday wording while the reference
answers the formal question it was rewritten from. Each set is written as answer_pairs.jsonl (key, question, reference, gold), the
input scripts/eval_article_evidence.py reads. The reference is the AIHub answer to the formal question, found by question text in the
upstream qa.jsonl files, as judge_answers.py does.

    uv run python scripts/build_article_heldout.py --out data/eval/reports/chat_pipeline/article-heldout
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parent.parent
QA_ROOT = Path.home() / "Documents/legal-rag/data/processed"

app = typer.Typer(add_completion=False)


def _references() -> dict[str, list[dict]]:
    found: dict[str, list[dict]] = collections.defaultdict(list)
    for path in QA_ROOT.glob("*/qa.jsonl"):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                record = json.loads(line)
                found[record["question"].strip()].append(record)
    return found


def _row(key: str, query: str, formal_query: str, gold: list[str], references: dict[str, list[dict]]) -> dict | None:
    candidates = references.get(formal_query.strip(), [])
    candidates = [record for record in candidates if record["document_id"] in gold] or candidates
    if not candidates:
        return None
    return {"key": key, "question": query, "reference": candidates[0]["answer"],
            "gold": sorted(gold, key=lambda document: "::statute::" not in document)}  # a statute first, as the script expects


@app.command()
def main(out: Path = typer.Option(..., help="Directory that receives formal/ and everyday/.")) -> None:
    import eval_chat_pipeline as chat  # scripts/ is on sys.path when this file runs as a script

    references = _references()
    used = {f"{name}:{item['id']}" for name, item in chat._sample(20, 11, 20)}
    sets: dict[str, list[dict | None]] = {"formal": [], "everyday": []}
    for name, item in chat._sample(50, 7, None):
        key = f"{name}:{item['id']}"
        if key not in used:
            sets["formal"].append(_row(key, item["query"], item["query"], item["relevant_document_ids"], references))
    for line in open(ROOT / "data/eval/everyday-150.jsonl", encoding="utf-8"):
        item = json.loads(line)
        sets["everyday"].append(_row(f"{item['dataset']}:{item['id']}", item["query"], item["formal_query"],
                                     item["relevant_document_ids"], references))
    for name, rows in sets.items():
        kept = [row for row in rows if row]
        (out / name).mkdir(parents=True, exist_ok=True)
        (out / name / "answer_pairs.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept), encoding="utf-8")
        print(f"{name}: {len(kept)} questions ({len(rows) - len(kept)} without a reference), "
              f"{sum('::statute::' in row['gold'][0] for row in kept)} with a statute gold document")


if __name__ == "__main__":
    app()
```

**Step 2: Build the two sets**

```bash
PYTHONPATH=src $PY scripts/build_article_heldout.py --out data/eval/reports/chat_pipeline/article-heldout
```

Expected:

```
formal: 169 questions (0 without a reference), 82 with a statute gold document
everyday: 150 questions (0 without a reference), 73 with a statute gold document
```

(21 of the everyday questions share an id with one of the 80; their wording differs from the formal question, which is what this set tests.)

**Step 3: Run both sets (about 35 minutes)**

The bars are criterion 3. They were written to `data/eval/reports/chat_pipeline/formal-2026-09-28-articles/statute-replay/heldout-bars.txt` before the first held-out run; write them again next to your results if you change anything.

```bash
R=data/eval/reports/chat_pipeline
for set in formal everyday; do
  PYTHONPATH=src $PY scripts/eval_article_evidence.py --run $R/article-heldout/$set --out $R/article-heldout/$set-coverage --root /home/layke/Documents/MyLLMAgent
done
python3 - <<'EOF'
import json
for name in ("formal", "everyday"):
    rows = [json.loads(line) for line in open(f"data/eval/reports/chat_pipeline/article-heldout/{name}-coverage/article_coverage.jsonl", encoding="utf-8")]
    steps = sorted(row["select_seconds"] for row in rows if row["statute"])
    print(name, "statute step time p90 %.2fs" % steps[int(0.9 * (len(steps) - 1))])
EOF
```

**Step 4: Read the result against criterion 3**

The two sets were run on 2026-09-29 in a scratch checkout that held this plan's code for Tasks 1 to 5 and for the two scripts (pool 30 plus 10, unchanged). Step 3 can be skipped in a fresh worktree when `diff -r` shows that the branch's `src/` and `scripts/eval_article_evidence.py` are identical to that checkout's; otherwise run it. Output:

```text
formal    statute questions: n=82  covered today 35  with article step 69  (+36 / -2)  extra cross-encoder pairs 147  step time 1.15s  evidence chars 7260 -> 3552
formal    other questions:   n=87  covered today 25  with article step 25  (+0 / -0)  extra cross-encoder pairs 29  step time 0.28s  evidence chars 6489 -> 6219
everyday  statute questions: n=73  covered today 30  with article step 54  (+27 / -3)  extra cross-encoder pairs 137  step time 1.06s  evidence chars 7291 -> 3542
everyday  other questions:   n=77  covered today 18  with article step 18  (+0 / -0)  extra cross-encoder pairs 30  step time 0.28s  evidence chars 6622 -> 6352
```

| Bar (`heldout-bars.txt`) | Formal, 169 questions | Everyday, 150 questions |
| --- | --- | --- |
| Statute gained minus lost at least 15% of the statute questions | +34 of 82 (41%; bar 13): pass | +24 of 73 (33%; bar 11): pass |
| Statute lost at most 2 | 2: pass, at the limit | 3: **fail** |
| Other questions lost at most 2 | 0: pass | 0: pass |
| Statute step time p90 at most 2.5 s | 1.77 s: pass | 1.72 s: pass |

Criterion 3 therefore fails on one condition of one set, by one question. The five losses are four distinct questions ("범칙금" is lost in both phrasings). Replaying them with the same code shows two causes:

- **Pool (both formal losses).** The window that holds the cited article is a fused candidate but lies outside the pool: reranker rank 69 for the 제21조 question (the article scores 1.00) and rank 48 for the 제58조 question (0.95). Today's evidence is credited for another window that only opens in that article; the opening text of that window scores 0.08 to 0.38.
- **Article scores (the three everyday losses).** For the 한센인 question the cited 제4조 of the gold statute scores 0.31, under that statute's own 제2조 (0.65) and 제7조 (0.64) and under another statute's 제4조 (0.94). For the 산업단지 question every article of the gold statute scores under 0.05 and the step shows other documents (best 0.57). For the everyday 범칙금 question the rewritten search query (`범칙금 벌금 임대차`) brings one window of the statute into the 100 candidates, its cited articles score 0.01 and 0.06, and court-decision fragments outscore them.

The gains are of the size seen in-sample (34%), so the step is not simply tuned to formal phrasing, but the pre-registered bar is missed. A wider pool might recover the two pool cases at more cross-encoder pairs; choosing it now would be a new tuning decision, and a rerun of the two sets would no longer be held-out for it.

If a bar fails, stop. Report which set failed and how, and do not go on to Task 7: the step is then tuned to formal phrasing or to these 80 questions. **On 2026-09-29 the everyday bar failed, so Task 7 has not been run; whether to run it anyway is a decision for the person who owns this plan.**

**Step 5: Commit**

```bash
git add scripts/build_article_heldout.py
git commit -m "eval: held-out formal and everyday question sets for the article step" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Answer-level A/B

**Gate:** start this task only after Task 6b, Step 4 says criterion 3 passed. On 2026-09-29 it did not (everyday set, 3 lost against at most 2), so this task has not been run.

The retrieval numbers say the article reaches the model. This task measures whether answers improve. Answers are sampled (temperature 0.2), so one run per arm is noisy: the `k_final` 8 rerun flipped 15 of 78 answers. The arms are therefore compared as means of three samples each on the 35 statute questions: three flag-on runs against three flag-off samples (the 2026-09-28 run plus two reruns). The 12 statute questions whose article the step newly shows are the test of the mechanism; the other 23 are the noise control, since their evidence is unchanged. Budget about four GPU hours, unattended.

**Files:**
- Modify: `scripts/eval_chat_pipeline.py` (record the documents of the shown evidence)
- Modify: `scripts/judge_answers.py` (use them)

**Step 1: Make the shown documents explicit**

Article-level evidence starts mid-chunk, so `judge_answers.py pairs` cannot place it by matching chunk prefixes. Apply:

```diff
diff --git a/scripts/eval_chat_pipeline.py b/scripts/eval_chat_pipeline.py
index cc00e71..2612bb8 100644
--- a/scripts/eval_chat_pipeline.py
+++ b/scripts/eval_chat_pipeline.py
@@ -183,4 +183,5 @@ def verify(
             if response.evidence is not None:
                 record["evidence"] = evidence(response.evidence)
+                record["evidence_documents"] = [item.document_id for item in response.evidence.items]
             if response.answer is not None:
                 # The strip note names the citations it removed; the guard would count those as cited.
diff --git a/scripts/judge_answers.py b/scripts/judge_answers.py
index f729367..96de7a8 100644
--- a/scripts/judge_answers.py
+++ b/scripts/judge_answers.py
@@ -93,5 +93,6 @@ def pairs(run: Path = typer.Option(..., exists=True, file_okay=False),
             typer.echo(f"no reference for {turn['key']}", err=True)
             continue
-        documents = [document_of(e) for e in turn.get("evidence", [])]
+        # Article-level evidence starts mid-chunk, so the chunk-prefix lookup cannot place it; newer runs record the documents.
+        documents = turn.get("evidence_documents") or [document_of(e) for e in turn.get("evidence", [])]
         out.append({"key": turn["key"], "dataset": turn["dataset"], "kind": turn["kind"], "question": question,
                     "reference": candidates[0]["answer"], "answer": turn.get("final_text", ""),
```

Check that old runs are unaffected:

```bash
mkdir -p /tmp/pairs-check && cp data/eval/reports/chat_pipeline/formal-2026-09-28-articles/verify.jsonl /tmp/pairs-check/
PYTHONPATH=src $PY scripts/judge_answers.py pairs --run /tmp/pairs-check
cmp /tmp/pairs-check/answer_pairs.jsonl data/eval/reports/chat_pipeline/formal-2026-09-28-articles/answer_pairs.jsonl && echo identical
```

Expected: `80 pairs, gold document in evidence for 47` and `identical`.

**Step 2: Commit**

```bash
git add scripts/eval_chat_pipeline.py scripts/judge_answers.py
git commit -m "eval: record the documents of the shown evidence" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

**Step 3: Run 1, all 80 questions with the step on (about 55 minutes), then judge (about 30 minutes)**

```bash
LIVE=/home/layke/Documents/MyLLMAgent
R=data/eval/reports/chat_pipeline
LEGAL_RAG__PATHS__ROOT=$LIVE PYTHONPATH=src $PY scripts/eval_chat_pipeline.py verify --out $R/articles-on-1 --overrides '{"rag": {"article_evidence": true}}'
curl -s localhost:11434/api/generate -d '{"model":"qwen3:4b","keep_alive":0}' >/dev/null
PYTHONPATH=src $PY scripts/judge_answers.py pairs --run $R/articles-on-1
PYTHONPATH=src $PY scripts/judge_answers.py judge --run $R/articles-on-1
curl -s localhost:11434/api/generate -d '{"model":"qwen3:8b","keep_alive":0}' >/dev/null
```

**Step 4: Runs 2 and 3, the 35 statute questions only (about 25 minutes each plus 15 to judge)**

```bash
python3 - <<'EOF'
import json
run = "data/eval/reports/chat_pipeline/formal-2026-09-28-articles/"
with open("data/eval/reports/chat_pipeline/statute-35.jsonl", "w", encoding="utf-8") as out:
    for line in open(run + "answer_pairs.jsonl", encoding="utf-8"):
        pair = json.loads(line)
        if "::statute::" in pair["gold"][0]:
            dataset, _, question_id = pair["key"].partition(":")
            out.write(json.dumps({"id": question_id, "query": pair["question"], "relevant_document_ids": pair["gold"], "dataset": dataset}, ensure_ascii=False) + "\n")
EOF
for n in 2 3; do
  LEGAL_RAG__PATHS__ROOT=$LIVE PYTHONPATH=src $PY scripts/eval_chat_pipeline.py verify --out $R/articles-on-$n --questions $R/statute-35.jsonl --overrides '{"rag": {"article_evidence": true}}'
  curl -s localhost:11434/api/generate -d '{"model":"qwen3:4b","keep_alive":0}' >/dev/null
  PYTHONPATH=src $PY scripts/judge_answers.py pairs --run $R/articles-on-$n
  PYTHONPATH=src $PY scripts/judge_answers.py judge --run $R/articles-on-$n
  curl -s localhost:11434/api/generate -d '{"model":"qwen3:8b","keep_alive":0}' >/dev/null
done
```

Expected: `pairs` prints `35 pairs, ...` for runs 2 and 3. If `pairs` reports `no reference for ...`, the question file's `dataset` or `id` does not match the sample files named in `SOURCES` in `judge_answers.py`.

**Step 5: Two flag-off reruns of the same 35 statute questions (required; about 40 minutes each with judging)**

Without them the flag-on runs would be compared with a single baseline run, and one run of 35 sampled answers cannot show a difference of three.

```bash
for n in 2 3; do
  LEGAL_RAG__PATHS__ROOT=$LIVE PYTHONPATH=src $PY scripts/eval_chat_pipeline.py verify --out $R/articles-off-$n --questions $R/statute-35.jsonl
  curl -s localhost:11434/api/generate -d '{"model":"qwen3:4b","keep_alive":0}' >/dev/null
  PYTHONPATH=src $PY scripts/judge_answers.py pairs --run $R/articles-off-$n
  PYTHONPATH=src $PY scripts/judge_answers.py judge --run $R/articles-off-$n
  curl -s localhost:11434/api/generate -d '{"model":"qwen3:8b","keep_alive":0}' >/dev/null
done
```

**Step 6: Compare against acceptance criterion 4**

`gained.json` (written in Task 6, Step 3) lists the 12 statute questions whose cited article the step newly shows. Save as `/tmp/compare_runs.py` (not committed):

```python
"""Correct answers per run and per arm, split into the statute questions the article step changes and the ones it does not.

    python compare_runs.py --gained gained.json --arm off RUN [RUN ...] --arm on RUN [RUN ...]
Runs are directories under data/eval/reports/chat_pipeline. gained.json lists the keys whose cited article the step newly shows
(Task 6 writes it). A run counts only the questions it contains, so statute-only runs report 35 statute questions and no others."""
import json
import sys
from pathlib import Path


def load(run):
    path = Path(run)
    pairs = {r["key"]: r for r in map(json.loads, open(path / "answer_pairs.jsonl", encoding="utf-8"))}
    grades = {r["key"]: r["grade"] for r in map(json.loads, open(path / "answer_judgments.jsonl", encoding="utf-8"))}
    verify = {r["key"]: r for r in map(json.loads, open(path / "verify.jsonl", encoding="utf-8"))}
    correct = {k: g is not None and g["covers"] == "yes" and not g["contradicts"] for k, g in grades.items()}
    return pairs, correct, verify


def counts(pairs, correct, verify, gained):
    keys = list(pairs)
    statute = [k for k in keys if "::statute::" in pairs[k]["gold"][0]]
    groups = {"statute": statute, "gained": [k for k in statute if k in gained], "unchanged": [k for k in statute if k not in gained],
              "other": [k for k in keys if k not in set(statute)]}
    row = {name: (sum(correct[k] for k in ks), len(ks)) for name, ks in groups.items()}
    row["final_ok"] = (sum(bool(verify[k]["final_ok"]) for k in keys), len(keys))
    row["flagged"] = (sum(1 for k in keys if verify[k]["attempts"] and (verify[k]["attempts"][0].get("raw_issues") or verify[k]["attempts"][0].get("issues"))), len(keys))
    row["seconds"] = sum(verify[k]["seconds"] for k in keys) / len(keys)
    return row


def show(name, row):
    parts = [f"{g} {row[g][0]:2d}/{row[g][1]:2d}" for g in ("statute", "gained", "unchanged", "other") if row[g][1]]
    print(f"{name:34s} " + "  ".join(parts) + f"  final_ok {row['final_ok'][0]}/{row['final_ok'][1]}  flagged {row['flagged'][0]}  turn {row['seconds']:.1f}s")


args = sys.argv[1:]
gained = set(json.load(open(args[args.index("--gained") + 1], encoding="utf-8")))
arms, current = {}, None
i = 0
while i < len(args):
    if args[i] == "--gained":
        i += 2
    elif args[i] == "--arm":
        current = args[i + 1]
        arms[current] = []
        i += 2
    else:
        arms[current].append(args[i])
        i += 1
means = {}
for arm, runs in arms.items():
    rows = []
    for run in runs:
        row = counts(*load(run), gained)
        rows.append(row)
        show(f"{arm}: {Path(run).name}", row)
    means[arm] = {g: sum(r[g][0] for r in rows) / len(rows) for g in ("statute", "gained", "unchanged")}
    print(f"{'':34s} {arm} mean over {len(rows)} runs: " + "  ".join(f"{g} {v:.1f}" for g, v in means[arm].items()))
if {"on", "off"} <= set(means):
    print("on minus off: " + "  ".join(f"{g} {means['on'][g] - means['off'][g]:+.1f}" for g in ("statute", "gained", "unchanged")))
```

```bash
python3 /tmp/compare_runs.py --gained $R/article-coverage/gained.json \
  --arm off $R/formal-2026-09-28-articles $R/articles-off-2 $R/articles-off-3 \
  --arm on $R/articles-on-1 $R/articles-on-2 $R/articles-on-3
```

Expected for the first flag-off line (the 2026-09-28 run): `statute 22/35  gained  4/12  unchanged 18/23  other 28/45  final_ok 76/80  flagged 10  turn 32.7s`. Read the result against criterion 4: `on minus off` for `statute` at least +3, `unchanged` within plus or minus 2, the first flag-on run's `other` at least 25, `final_ok` at least 75, `flagged` at most 13, `turn` not above 32.7 s. Write the numbers into the doc (Task 8) whether or not they pass.

---

### Task 8: Document the result and decide the setting

**Files:**
- Modify: `docs/legal-rag-backend.md` (after the `k_final` 5 to 8 section, and the "Main loss" bullet)
- Modify: `configs/local.yaml` (only if criteria 2, 3 and 4 pass)

**Step 1: Correct the answer-accuracy conclusion**

In `docs/legal-rag-backend.md` replace

```
- **Main loss:** retrieval. With the gold document in evidence, 70% of answers match the
  reference; without it, 52%.
```

with

```
- **Where answers are lost:** of the 30 non-correct answers, 15 come from questions that name
  no specific law or case (the reference is one source's statement), 9 lack the key passage in
  the shown evidence (7 are statute questions), 5 are model errors and 1 is a judge error. The
  70% against 52% gap by gold document is therefore mostly weak references, not retrieval alone.
  One reader, uncalibrated; labels in `data/eval/reports/chat_pipeline/formal-2026-09-28-articles/failure-breakdown/`.
```

**Step 2: Add the section**

After the `### More evidence per answer (k_final 5 → 8), 2026-09-28: not adopted` section add `### Article-level evidence for statute windows, 2026-09-29` with: the failure table from "Why this change" above, the pool table, the Task 6 output, the Task 7 comparison lines and the decision. Keep it to the numbers and what they mean; state that `k_final` 8, `k_fused` 150, neighbour chunks and re-chunking were tested and are not needed.

**Step 3: Set the flag only if the criteria pass**

If criteria 2, 3 and 4 all pass, add to `configs/local.yaml` (the file has no `rag:` section yet):

```yaml
rag:
  # Statute windows hold several articles and only their first 1,500 characters reach the model, while
  # the cross-encoder reads about 900. Scoring each article instead put the cited article in the
  # evidence for 28 of 35 statute questions against 16, for ~1.3 s more per statute question
  # (docs/legal-rag-backend.md, 2026-09-29).
  article_evidence: true
```

If either fails, leave the file as it is.

**Step 4: Run the suite and commit**

```bash
PYTHONPATH=src $PY -m pytest tests/legal_rag -q -p no:cacheprovider
git add docs/legal-rag-backend.md configs/local.yaml
git commit -m "docs: article-level statute evidence, result and setting" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

Expected: `562 passed, 3 skipped`.

---

## Not in this plan

- **Evidence-sufficiency check.** The verifier checks citations against sources, not that the shown sources answer the question: it passed 8 of the 9 evidence-gap answers, including one that applied an unrelated provision's deadline. A check that lets the model say "the evidence does not contain this" is a separate design.
- **The Flask `/chat` path** has the same head-of-window display (5 documents of 1,200 characters). The selector could serve it, but that path needs its own measurement; its excerpt-selector test on the IP and mixed development sets was a null.
- **The 15 question-mismatch failures** are an evaluation-set problem: generic questions scored against one court's statement. Fixing that means rewriting or dropping those questions, or scoring with a rubric.
- **Candidate recall for the 7 uncovered statute questions.** Three name no law; four need a first-stage change, and the tested options (`k_fused` 150, deeper leg floor) recover at most 2.

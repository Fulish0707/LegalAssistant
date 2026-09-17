# Graded Relevance Evaluation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace exact-gold retrieval metrics with LLM-judged graded relevance over a pooled, append-only judgment store, validated against 50 human labels.

**Architecture:** A pool builder retrieves the 200-case ip-v1 target in dense/sparse/hybrid mode, records each mode's ranking to depth 100, and writes the union of top-20 chunks plus gold chunks to `pool.jsonl`. A judge appends grades (0/1/2) to `judgments.jsonl`, keyed by model and prompt hash. A stratified calibration CSV, led by a searchable case number, collects human grades; an agreement script checks the judge; a graded-metrics module scores each recorded run.

**Tech Stack:** Python 3.12, typer, pydantic, Ollama (`qwen3:8b` judge, `bge-m3` embeddings), Qdrant server, pytest.

**Spec:** `docs/superpowers/specs/2026-09-17-graded-relevance-eval-design.md`

**Conventions:** run tests with `uv run pytest`. New modules live in `src/legal_rag/evaluation/`, tests in `tests/legal_rag/evaluation/`. Every writer refuses to overwrite (`open('x')` / `mkdir(exist_ok=False)`) except the judgment store, which only appends.

**Output layout** (`data/eval/relevance/ip-v1-pilot-200/`):

```
pool.json            provenance: target sha, settings hash, filters, depths
pool.jsonl           one row per (case, chunk) to judge
runs/<mode>.jsonl    one row per case: ranked [{chunk_id, document_id}] to depth 100
judgments.jsonl      append-only store
calibration.csv      human labels (UTF-8 BOM for Excel)
calibration.json     pair ids and strata — kept out of the CSV to avoid biasing the labeller
```

---

### Task 1: Searchable case labels

**Files:**
- Create: `src/legal_rag/evaluation/case_labels.py`
- Test: `tests/legal_rag/evaluation/test_case_labels.py`

**Step 1: Write the failing test**

```python
from legal_rag.evaluation.case_labels import case_label


def chunk(document_id, document_type, *, title='', text='본문', authority=None, date=None):
    return dict(document_id=document_id, document_type=document_type, title=title, text=text,
                authority=authority, date=date)


def test_court_judgment_splits_court_from_number():
    label = case_label(chunk('서울중앙지방법원-2016노4825', 'judgment', authority='서울중앙지방법원'))
    assert label == {'case_number': '2016노4825', 'authority': '서울중앙지방법원',
                     'search_query': '서울중앙지방법원 2016노4825'}


def test_patent_court_judgment():
    assert case_label(chunk('특허법원-2010허8443', 'judgment'))['search_query'] == '특허법원 2010허8443'


def test_interpretation_keeps_agenda_number():
    label = case_label(chunk('법제처-법령해석사례-법제처-09-0381', 'legal_interpretation', authority='법제처'))
    assert label['case_number'] == '09-0381'
    assert label['search_query'] == '법제처 09-0381'


def test_statute_uses_law_name_and_opening_article():
    label = case_label(chunk('statute::상표법::2024-08-07 00:00:00', 'statute', title='상표법',
                             text='제73조의2(상표등록의 취소심판)\n...'))
    assert label['case_number'] == '상표법 제73조의2'
    assert label['search_query'] == '상표법 제73조의2'


def test_statute_without_article_falls_back_to_law_name():
    label = case_label(chunk('statute::상표법::2024', 'statute', title='상표법', text='부칙 ...'))
    assert label['case_number'] == '상표법'


def test_other_types_use_document_id_never_blank():
    label = case_label(chunk('2011101003196', 'patent_trial_decision', authority='특허심판원'))
    assert label == {'case_number': '2011101003196', 'authority': '특허심판원',
                     'search_query': '특허심판원 2011101003196'}
    assert case_label(chunk('2022기정1485', 'decision'))['search_query'] == '2022기정1485'
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/legal_rag/evaluation/test_case_labels.py -v`
Expected: FAIL, `ModuleNotFoundError: legal_rag.evaluation.case_labels`

**Step 3: Implement**

```python
"""Human-searchable case numbers for pooled chunks.

The corpus leaves ``metadata.case_number`` empty, but ``document_id`` carries the number.
A label always falls back to ``document_id`` so a reviewer never gets a blank cell.
"""
from __future__ import annotations

import re

_COURT_ID = re.compile(r'^(?P<court>[^-]*법원(?:[^-]*지원)?)-(?P<number>.+)$')
_AGENDA = re.compile(r'(\d{2}-\d{4})$')
_ARTICLE = re.compile(r'^\s*(제\d+조(?:의\d+)?)')


def case_label(chunk: dict) -> dict:
    document_id, document_type = chunk['document_id'], chunk['document_type']
    authority = chunk.get('authority')
    if document_type == 'statute':
        law = chunk.get('title') or document_id.split('::')[1]
        article = _ARTICLE.match(chunk.get('text') or '')
        number = f'{law} {article.group(1)}' if article else law
        return {'case_number': number, 'authority': authority, 'search_query': number}
    court = _COURT_ID.match(document_id)
    if court:
        authority = authority or court['court']
        number = court['number']
    elif document_type == 'legal_interpretation' and _AGENDA.search(document_id):
        number = _AGENDA.search(document_id).group(1)
    else:
        number = document_id
    query = f'{authority} {number}' if authority else number
    return {'case_number': number, 'authority': authority, 'search_query': query}
```

**Step 4: Run to verify pass**

Run: `uv run pytest tests/legal_rag/evaluation/test_case_labels.py -v`
Expected: 6 passed

**Step 5: Commit**

```bash
git add src/legal_rag/evaluation/case_labels.py tests/legal_rag/evaluation/test_case_labels.py
git commit -m "feat: derive searchable case numbers for relevance review"
```

---

### Task 2: Append-only judgment store

**Files:**
- Create: `src/legal_rag/evaluation/judgment_store.py`
- Test: `tests/legal_rag/evaluation/test_judgment_store.py`

**Step 1: Write the failing test**

```python
from legal_rag.evaluation.judgment_store import JudgeKey, JudgmentStore

KEY = JudgeKey(model='ollama:qwen3:8b', prompt_sha256='p1')


def test_append_then_reload(tmp_path):
    store = JudgmentStore(tmp_path / 'judgments.jsonl')
    store.append(KEY, case_id='c1', chunk_id='k1', document_id='d1', grade=2, reason='r')
    reloaded = JudgmentStore(tmp_path / 'judgments.jsonl')
    assert reloaded.grades(KEY) == {('c1', 'k1'): 2}
    assert reloaded.has(KEY, 'c1', 'k1')
    assert reloaded.documents(KEY) == {('c1', 'k1'): 'd1'}


def test_other_model_or_prompt_is_a_separate_judgment(tmp_path):
    store = JudgmentStore(tmp_path / 'judgments.jsonl')
    store.append(KEY, case_id='c1', chunk_id='k1', document_id='d1', grade=2, reason='')
    other = JudgeKey(model='ollama:qwen3:8b', prompt_sha256='p2')
    assert not store.has(other, 'c1', 'k1')
    store.append(other, case_id='c1', chunk_id='k1', document_id='d1', grade=0, reason='')
    assert store.grades(KEY) == {('c1', 'k1'): 2}
    assert store.grades(other) == {('c1', 'k1'): 0}


def test_null_grade_is_recorded_but_not_a_grade(tmp_path):
    store = JudgmentStore(tmp_path / 'judgments.jsonl')
    store.append(KEY, case_id='c1', chunk_id='k1', document_id='d1', grade=None, reason='', raw='garbage')
    assert store.has(KEY, 'c1', 'k1')          # never re-judged on resume
    assert store.grades(KEY) == {}
    assert store.failures(KEY) == [('c1', 'k1')]


def test_duplicate_append_is_refused(tmp_path):
    import pytest
    store = JudgmentStore(tmp_path / 'judgments.jsonl')
    store.append(KEY, case_id='c1', chunk_id='k1', document_id='d1', grade=1, reason='')
    with pytest.raises(ValueError, match='already judged'):
        store.append(KEY, case_id='c1', chunk_id='k1', document_id='d1', grade=2, reason='')
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/legal_rag/evaluation/test_judgment_store.py -v`
Expected: FAIL, module not found

**Step 3: Implement**

```python
"""Append-only relevance judgments.

A judgment is identified by (judge model, prompt hash, case, chunk) and is never rewritten.
Changing the model or prompt creates new keys, so judgments from different judges coexist
and are never mixed silently.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class JudgeKey:
    model: str
    prompt_sha256: str


class JudgmentStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._rows: dict[tuple, dict] = {}
        if self.path.exists():
            with self.path.open(encoding='utf-8') as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        self._rows[self._key(JudgeKey(row['judge_model'], row['prompt_sha256']),
                                             row['case_id'], row['chunk_id'])] = row

    @staticmethod
    def _key(judge: JudgeKey, case_id: str, chunk_id: str) -> tuple:
        return (judge.model, judge.prompt_sha256, case_id, chunk_id)

    def has(self, judge: JudgeKey, case_id: str, chunk_id: str) -> bool:
        return self._key(judge, case_id, chunk_id) in self._rows

    def append(self, judge: JudgeKey, *, case_id: str, chunk_id: str, document_id: str,
               grade: int | None, reason: str, raw: str | None = None) -> None:
        key = self._key(judge, case_id, chunk_id)
        if key in self._rows:
            raise ValueError(f'already judged: {key}')
        row = {'case_id': case_id, 'chunk_id': chunk_id, 'document_id': document_id, 'grade': grade,
               'reason': reason, 'judge_model': judge.model, 'prompt_sha256': judge.prompt_sha256,
               'judged_at': datetime.now(timezone.utc).isoformat(), 'raw': raw}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            handle.flush()
        self._rows[key] = row

    def _for(self, judge: JudgeKey):
        return (row for (model, prompt, _, _), row in self._rows.items()
                if (model, prompt) == (judge.model, judge.prompt_sha256))

    def grades(self, judge: JudgeKey) -> dict[tuple[str, str], int]:
        return {(row['case_id'], row['chunk_id']): row['grade'] for row in self._for(judge)
                if row['grade'] is not None}

    def documents(self, judge: JudgeKey) -> dict[tuple[str, str], str]:
        return {(row['case_id'], row['chunk_id']): row['document_id'] for row in self._for(judge)}

    def failures(self, judge: JudgeKey) -> list[tuple[str, str]]:
        return sorted((row['case_id'], row['chunk_id']) for row in self._for(judge) if row['grade'] is None)
```

**Step 4: Run to verify pass**

Run: `uv run pytest tests/legal_rag/evaluation/test_judgment_store.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add src/legal_rag/evaluation/judgment_store.py tests/legal_rag/evaluation/test_judgment_store.py
git commit -m "feat: append-only relevance judgment store"
```

---

### Task 3: LLM relevance judge

**Files:**
- Create: `src/legal_rag/evaluation/relevance_judge.py`
- Test: `tests/legal_rag/evaluation/test_relevance_judge.py`

**Step 1: Write the failing test**

```python
from legal_rag.evaluation.judgment_store import JudgmentStore
from legal_rag.evaluation.relevance_judge import RelevanceGrade, judge_key, judge_pool
from legal_rag.llm.base import ModelResponse, StructuredOutputError


class ScriptedModel:
    name = 'scripted'

    def __init__(self, replies):
        self.replies, self.messages = list(replies), []

    def generate(self, messages, *, response_schema=None, temperature=0.2, max_tokens=1024):
        self.messages.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ModelResponse(text='', parsed=RelevanceGrade(**reply))


def pair(chunk_id='k1', **extra):
    return dict(case_id='c1', chunk_id=chunk_id, document_id='d1', document_type='judgment',
                title='상표법위반', text='부정경쟁의 목적이란 …', question='부정경쟁의 목적은?',
                reference_answer='신용을 이용하여 부당한 이익을 얻을 목적', ranks={'hybrid': 3},
                is_gold_chunk=True, **extra)


def test_grades_are_appended_and_resume_skips_judged(tmp_path):
    store = JudgmentStore(tmp_path / 'j.jsonl')
    model = ScriptedModel([{'grade': 2, 'reason': '법리 명시'}])
    assert judge_pool([pair()], model, store) == {'judged': 1, 'skipped': 0, 'failed': 0}
    again = judge_pool([pair()], ScriptedModel([]), store)
    assert again == {'judged': 0, 'skipped': 1, 'failed': 0}
    assert store.grades(judge_key(model)) == {('c1', 'k1'): 2}


def test_prompt_hides_rank_mode_and_gold_status(tmp_path):
    model = ScriptedModel([{'grade': 1, 'reason': ''}])
    judge_pool([pair()], model, JudgmentStore(tmp_path / 'j.jsonl'))
    prompt = ' '.join(message.content for message in model.messages[0])
    assert '부정경쟁의 목적은?' in prompt and '부당한 이익' in prompt and '상표법위반' in prompt
    for leaked in ('hybrid', 'gold', 'rank', 'is_gold_chunk'):
        assert leaked not in prompt


def test_malformed_output_retries_once_then_records_null_not_zero(tmp_path):
    store = JudgmentStore(tmp_path / 'j.jsonl')
    bad = StructuredOutputError('bad', 'not json')
    model = ScriptedModel([bad, bad])
    assert judge_pool([pair()], model, store) == {'judged': 0, 'skipped': 0, 'failed': 1}
    assert len(model.messages) == 2
    assert store.grades(judge_key(model)) == {}
    assert store.failures(judge_key(model)) == [('c1', 'k1')]


def test_retry_success_counts_as_judged(tmp_path):
    model = ScriptedModel([StructuredOutputError('bad', 'x'), {'grade': 0, 'reason': ''}])
    assert judge_pool([pair()], model, JudgmentStore(tmp_path / 'j.jsonl'))['judged'] == 1


def test_long_chunks_are_truncated_in_the_prompt(tmp_path):
    model = ScriptedModel([{'grade': 0, 'reason': ''}])
    judge_pool([pair(text='가' * 10_000)], model, JudgmentStore(tmp_path / 'j.jsonl'))
    assert '가' * 3000 not in ' '.join(m.content for m in model.messages[0])
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/legal_rag/evaluation/test_relevance_judge.py -v`
Expected: FAIL, module not found

**Step 3: Implement**

```python
"""Grade pooled (question, chunk) pairs with a chat model.

The prompt shows the question, the reference answer and the chunk only — never the
retrieval mode, rank, or whether the chunk belongs to the gold document.
"""
from __future__ import annotations

import hashlib
import json
from typing import Callable, Iterable, Literal

from pydantic import BaseModel

from legal_rag.llm.base import ChatModel, Message, StructuredOutputError

from .judgment_store import JudgeKey, JudgmentStore

MAX_CHUNK_CHARS = 2500
MAX_ANSWER_CHARS = 800

SYSTEM = """당신은 한국 법률 검색 평가자입니다. 질문, 참조 답변, 검색된 문서 조각이 주어집니다.
문서 조각이 참조 답변의 근거로 얼마나 쓸모 있는지 등급을 매기십시오.

2 = 참조 답변이 의존하는 법리·규정·판단 기준을 이 조각이 직접 담고 있어, 이 조각만으로 답변을 뒷받침할 수 있음.
    참조 답변을 작성할 때 쓰인 문서가 아니더라도 같은 법리를 명시하면 2입니다.
1 = 같은 법리나 조문을 다루지만 이 조각만으로는 답변을 뒷받침하기에 부족함.
0 = 무관하거나 용어만 겹침.

JSON으로만 답하십시오: {"grade": 0|1|2, "reason": "한 문장 근거"}"""

USER = """[질문]
{question}

[참조 답변]
{answer}

[문서 조각] ({document_type}) {title}
{text}"""


class RelevanceGrade(BaseModel):
    grade: Literal[0, 1, 2]
    reason: str


PROMPT_SHA256 = hashlib.sha256(
    json.dumps([SYSTEM, USER, RelevanceGrade.model_json_schema(), MAX_CHUNK_CHARS, MAX_ANSWER_CHARS],
               ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def judge_key(model: ChatModel) -> JudgeKey:
    return JudgeKey(model=model.name, prompt_sha256=PROMPT_SHA256)


def _messages(pair: dict) -> list[Message]:
    user = USER.format(question=pair['question'], answer=(pair['reference_answer'] or '')[:MAX_ANSWER_CHARS],
                       document_type=pair['document_type'], title=pair.get('title') or '',
                       text=pair['text'][:MAX_CHUNK_CHARS])
    return [Message(role='system', content=SYSTEM), Message(role='user', content=user)]


def judge_pool(pairs: Iterable[dict], model: ChatModel, store: JudgmentStore, *,
               progress: Callable[[str], None] | None = None) -> dict[str, int]:
    key = judge_key(model)
    counts = {'judged': 0, 'skipped': 0, 'failed': 0}
    pairs = list(pairs)
    for number, pair in enumerate(pairs, 1):
        if store.has(key, pair['case_id'], pair['chunk_id']):
            counts['skipped'] += 1
            continue
        grade, reason, raw = None, '', None
        for _attempt in range(2):
            try:
                parsed = model.generate(_messages(pair), response_schema=RelevanceGrade,
                                        temperature=0.0, max_tokens=256).parsed
                grade, reason = parsed.grade, parsed.reason
                break
            except StructuredOutputError as exc:
                raw = exc.text
        store.append(key, case_id=pair['case_id'], chunk_id=pair['chunk_id'], document_id=pair['document_id'],
                     grade=grade, reason=reason, raw=None if grade is not None else raw)
        counts['judged' if grade is not None else 'failed'] += 1
        if progress and number % 25 == 0:
            progress(f'{number}/{len(pairs)} {counts}')
    return counts
```

**Step 4: Run to verify pass**

Run: `uv run pytest tests/legal_rag/evaluation/test_relevance_judge.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add src/legal_rag/evaluation/relevance_judge.py tests/legal_rag/evaluation/test_relevance_judge.py
git commit -m "feat: LLM relevance judge with blind prompt and null-on-failure"
```

---

### Task 4: Pooling and run records

**Files:**
- Create: `src/legal_rag/evaluation/relevance_pool.py`
- Modify: `src/legal_rag/evaluation/diagnostic_runner.py` — extract the depth-raising settings copy into `deepen_retrieval(settings, deep_k)` and call it where `scoped` is built.
- Test: `tests/legal_rag/evaluation/test_relevance_pool.py`

**Step 1: Write the failing test**

```python
import json

import pytest

from legal_rag.evaluation.relevance_pool import build_pool, read_jsonl, write_pool
from legal_rag.schemas import Chunk, ChunkMetadata, DocumentType, ScoredChunk


def make_chunk(chunk_id, document_id=None, text='본문'):
    return Chunk(id=chunk_id, document_id=document_id or chunk_id.split('#')[0], document_type=DocumentType.judgment,
                 path='p', order=0, text=text, title='제목',
                 metadata=ChunkMetadata(dataset='d', corpus_version='ip-v1', authority='특허법원'))


class ListRetriever:
    def __init__(self, ranking):
        self.ranking, self.calls = ranking, []

    def retrieve(self, query, k, filters=None):
        self.calls.append((query, k, filters))
        return [ScoredChunk(chunk=make_chunk(c), score=1.0, retriever='fused') for c in self.ranking[:k]]


CASES = [dict(case_id='c1', question='질문', answer='답변', document_id='gold', legal_domain='intellectual_property')]


def test_pool_is_union_of_top_depth_plus_gold_with_each_modes_rank():
    retrievers = {'dense': ListRetriever(['a#0', 'b#0', 'c#0']), 'sparse': ListRetriever(['b#0', 'd#0'])}
    pool, runs = build_pool(CASES, retrievers, gold_chunks={'gold': [make_chunk('gold#0')]},
                            pool_depth=2, run_depth=3)
    rows = {row['chunk_id']: row for row in pool}
    assert set(rows) == {'a#0', 'b#0', 'd#0', 'gold#0'}           # c#0 is dense rank 3 > pool_depth
    assert rows['b#0']['ranks'] == {'dense': 2, 'sparse': 1}
    assert rows['gold#0']['ranks'] == {} and rows['gold#0']['is_gold_chunk']
    assert rows['a#0']['question'] == '질문' and rows['a#0']['reference_answer'] == '답변'
    assert rows['a#0']['search_query'] == '특허법원 a'
    assert runs['dense'] == [{'case_id': 'c1', 'ranked': [
        {'chunk_id': 'a#0', 'document_id': 'a'}, {'chunk_id': 'b#0', 'document_id': 'b'},
        {'chunk_id': 'c#0', 'document_id': 'c'}]}]


def test_retrieval_uses_run_depth_and_the_domain_filter():
    dense = ListRetriever(['a#0'])
    build_pool(CASES, {'dense': dense}, gold_chunks={}, pool_depth=1, run_depth=100)
    query, k, filters = dense.calls[0]
    assert (query, k, filters.legal_domain) == ('질문', 100, 'intellectual_property')


def test_pool_depth_cannot_exceed_run_depth():
    with pytest.raises(ValueError):
        build_pool(CASES, {'dense': ListRetriever([])}, gold_chunks={}, pool_depth=20, run_depth=10)


def test_write_refuses_existing_directory(tmp_path):
    pool, runs = build_pool(CASES, {'dense': ListRetriever(['a#0'])}, gold_chunks={}, pool_depth=1, run_depth=1)
    write_pool(tmp_path / 'out', pool, runs, {'target_sha256': 'x'})
    assert read_jsonl(tmp_path / 'out' / 'pool.jsonl')[0]['chunk_id'] == 'a#0'
    assert json.loads((tmp_path / 'out' / 'pool.json').read_text())['pairs'] == 1
    with pytest.raises(FileExistsError):
        write_pool(tmp_path / 'out', pool, runs, {})
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/legal_rag/evaluation/test_relevance_pool.py -v`
Expected: FAIL, module not found

**Step 3: Implement `relevance_pool.py`**

```python
"""Pool retrieved chunks from several modes for relevance judging, and record rankings."""
from __future__ import annotations

import json
from pathlib import Path

from legal_rag.retrieval.hybrid import Retriever
from legal_rag.schemas import Chunk, RetrievalFilters

from .case_labels import case_label


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _pair(case: dict, chunk: Chunk) -> dict:
    fields = dict(document_id=chunk.document_id, document_type=str(chunk.document_type), title=chunk.title,
                  text=chunk.text, authority=chunk.metadata.authority or chunk.metadata.court,
                  date=chunk.metadata.original.get('date'))
    return dict(case_id=case['case_id'], chunk_id=chunk.id, question=case['question'],
                reference_answer=case.get('answer'), **fields, **case_label(fields),
                ranks={}, is_gold_chunk=chunk.document_id == case['document_id'])


def build_pool(cases: list[dict], retrievers: dict[str, Retriever], *, gold_chunks: dict[str, list[Chunk]],
               pool_depth: int, run_depth: int) -> tuple[list[dict], dict[str, list[dict]]]:
    if not 1 <= pool_depth <= run_depth:
        raise ValueError('require 1 <= pool_depth <= run_depth')
    pool: dict[tuple[str, str], dict] = {}
    runs: dict[str, list[dict]] = {mode: [] for mode in retrievers}
    for case in cases:
        filters = RetrievalFilters(legal_domain=case['legal_domain'])
        for mode, retriever in retrievers.items():
            hits = retriever.retrieve(case['question'], run_depth, filters)[:run_depth]
            runs[mode].append({'case_id': case['case_id'], 'ranked': [
                {'chunk_id': hit.chunk.id, 'document_id': hit.chunk.document_id} for hit in hits]})
            for rank, hit in enumerate(hits[:pool_depth], 1):
                row = pool.setdefault((case['case_id'], hit.chunk.id), _pair(case, hit.chunk))
                row['ranks'][mode] = rank
        # Gold chunks are judged whatever their rank, to check the judge against the source.
        for chunk in gold_chunks.get(case['document_id'], []):
            pool.setdefault((case['case_id'], chunk.id), _pair(case, chunk))
    return list(pool.values()), runs


def write_pool(output: Path, pool: list[dict], runs: dict[str, list[dict]], provenance: dict) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'runs').mkdir()
    for name, rows in [('pool.jsonl', pool), *((f'runs/{mode}.jsonl', rows) for mode, rows in runs.items())]:
        with (output / name).open('x', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    (output / 'pool.json').write_text(json.dumps(
        {**provenance, 'pairs': len(pool), 'modes': list(runs)}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return output
```

**Step 4: Refactor `diagnostic_runner.py`**

Add, above `run_retrieval_diagnostics`:

```python
def deepen_retrieval(settings: Settings, deep_k: int) -> Settings:
    """Raise candidate depths on a copy; fusion weights, models and reranker are unchanged."""
    return settings.model_copy(update={'retrieval': settings.retrieval.model_copy(update={
        'k_dense': max(deep_k, settings.retrieval.k_dense),
        'k_sparse': max(deep_k, settings.retrieval.k_sparse),
        'k_fused': max(deep_k, settings.retrieval.k_fused),
    })})
```

and replace the inline `scoped = settings.model_copy(...)` block with `scoped = deepen_retrieval(settings, deep_k)`, keeping its comment.

**Step 5: Run to verify pass**

Run: `uv run pytest tests/legal_rag/evaluation -v`
Expected: all pass, including the existing diagnostic tests

**Step 6: Commit**

```bash
git add src/legal_rag/evaluation/relevance_pool.py src/legal_rag/evaluation/diagnostic_runner.py tests/legal_rag/evaluation/test_relevance_pool.py
git commit -m "feat: pool multi-mode retrieval for relevance judging"
```

---

### Task 5: Pool builder script (live)

**Files:**
- Create: `scripts/build_relevance_pool.py`

No unit test: the script only wires Task 4 to the live factory and is verified by running it.

**Step 1: Write the script**

```python
#!/usr/bin/env python
"""Retrieve a diagnostic target in several modes and pool chunks for relevance judging.

Example:
    uv run python scripts/build_relevance_pool.py --profile local --corpus-version ip-v1 \
      --domain intellectual_property \
      --target data/eval/reports/retrieval_diagnostics/ip-v1/intellectual_property/hybrid-seed42-final/target.jsonl \
      --output data/eval/relevance/ip-v1-pilot-200
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import typer

from legal_rag.cli import settings_from_options
from legal_rag.evaluation.diagnostic_runner import _gold_chunk_index, _qa_path, deepen_retrieval, load_diagnostic_rows
from legal_rag.evaluation.relevance_pool import build_pool, read_jsonl, write_pool
from legal_rag.factory import build_hybrid_retriever, close_legs, select_legs
from legal_rag.ingestion.corpus import Corpus

app = typer.Typer(add_completion=False)


@app.command()
def main(
    domain: str = typer.Option(...),
    target: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(...),
    profile: str = typer.Option(None),
    corpus_version: str = typer.Option(None),
    qa_file: Path = typer.Option(None, exists=True, dir_okay=False),
    modes: list[str] = typer.Option(['dense', 'sparse', 'hybrid'], '--mode'),
    pool_depth: int = typer.Option(20, min=1),
    run_depth: int = typer.Option(100, min=1),
    max_gold_chunks: int = typer.Option(10, min=1),
) -> None:
    if output.exists():
        raise typer.BadParameter(f'{output} exists; choose a new --output')
    settings = deepen_retrieval(settings_from_options(profile=profile, corpus_version=corpus_version), run_depth)
    qa_file = _qa_path(settings, qa_file)
    cases, _, provenance = load_diagnostic_rows(qa_file, domain=domain, dataset=target)
    target_ids = {row['id'] for row in read_jsonl(target)}
    if {case['case_id'] for case in cases} != target_ids:
        raise typer.BadParameter('QA rows resolved from --target do not reproduce its case ids')
    typer.echo(f'{len(cases)} cases; scanning gold chunks', err=True)
    gold, _ = _gold_chunk_index(Corpus.load(settings.processed_corpus_dir()), {c['document_id'] for c in cases},
                                max_gold_chunks=max_gold_chunks, preview_chars=10**9)
    hybrid = build_hybrid_retriever(settings)
    try:
        retrievers = {mode: select_legs(hybrid, mode) for mode in modes}
        pool, runs = build_pool(cases, retrievers, gold_chunks=gold, pool_depth=pool_depth, run_depth=run_depth)
    finally:
        close_legs(hybrid)
    write_pool(output, pool, runs, {
        'target': str(target.resolve()), 'target_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'qa_file': provenance['qa_file'], 'corpus_version': settings.corpus.version,
        'retrieval_corpora': settings.retrieval.corpora or [settings.corpus.version],
        'filters': {'legal_domain': domain}, 'pool_depth': pool_depth, 'run_depth': run_depth,
        'reranker': settings.rerank.backend, 'settings_hash': settings.hash()})
    typer.echo(f'wrote {output}: {len(pool)} pairs')


if __name__ == '__main__':
    app()
```

**Step 2: Check prerequisites**

Run: `ollama ps` after a test embed (see docs) — PROCESSOR must be `100% GPU`. Run: `curl -s localhost:6333/collections | head -c 300` — Qdrant must answer.

**Step 3: Run it**

```bash
uv run python scripts/build_relevance_pool.py --profile local --corpus-version ip-v1 \
  --domain intellectual_property \
  --target data/eval/reports/retrieval_diagnostics/ip-v1/intellectual_property/hybrid-seed42-final/target.jsonl \
  --output data/eval/relevance/ip-v1-pilot-200
```

Expected: `wrote … : N pairs` with N roughly 5,000–9,000.

**Step 4: Sanity-check reproduction against the diagnostic run**

```bash
uv run python - <<'EOF'
import json
D='data/eval/reports/retrieval_diagnostics/ip-v1/intellectual_property/hybrid-seed42-final/evaluated.jsonl'
gold={}
for l in open('data/eval/reports/retrieval_diagnostics/ip-v1/intellectual_property/hybrid-seed42-final/target.jsonl'):
    r=json.loads(l); gold[r['id']]=r['relevant_document_ids'][0]
old={json.loads(l)['case_id']:json.loads(l)['gold_rank'] for l in open(D)}
new={}
for l in open('data/eval/relevance/ip-v1-pilot-200/runs/hybrid.jsonl'):
    r=json.loads(l)
    new[r['case_id']]=next((i for i,h in enumerate(r['ranked'],1) if h['document_id']==gold[r['case_id']]),None)
print('hybrid gold ranks identical:', sum(old[c]==new[c] for c in old), '/', len(old))
EOF
```

Expected: `200 / 200`. A mismatch means the settings differ from the diagnostic run; stop and investigate before judging.

**Step 5: Commit**

```bash
git add scripts/build_relevance_pool.py
git commit -m "feat: build relevance pool from a diagnostic target"
```

---

### Task 6: Calibration sampling, CSV, and agreement

**Files:**
- Create: `src/legal_rag/evaluation/calibration.py`
- Test: `tests/legal_rag/evaluation/test_calibration.py`

**Step 1: Write the failing test**

```python
import csv
import json

import pytest

from legal_rag.evaluation.calibration import (
    STRATA, agreement, read_human_grades, sample_calibration, write_calibration)


def pool(n_cases=30):
    rows = []
    for case in range(n_cases):
        for rank in range(1, 21):
            rows.append(dict(case_id=f'c{case}', chunk_id=f'c{case}-k{rank}', document_id=f'd{rank}',
                             document_type='judgment', title='t', text='본문', question='질문',
                             reference_answer='답', case_number=f'2020허{rank}', authority='특허법원',
                             search_query=f'특허법원 2020허{rank}', date='2020.01.01',
                             ranks={'dense': rank}, is_gold_chunk=False))
        rows.append(dict(rows[-1], chunk_id=f'c{case}-gold', ranks={}, is_gold_chunk=True))
    return rows


def test_sample_meets_quotas_and_is_deterministic():
    sample = sample_calibration(pool(), seed=42)
    counts = {name: sum(row['stratum'] == name for row in sample) for name in STRATA}
    assert counts == {'gold': 10, 'rank_1_3': 15, 'rank_4_10': 15, 'rank_11_20': 10}
    assert sample == sample_calibration(list(reversed(pool())), seed=42)
    assert len({(row['case_id'], row['chunk_id']) for row in sample}) == 50


def test_csv_leads_with_searchable_case_number_and_hides_strata(tmp_path):
    sample = sample_calibration(pool(), seed=42)
    write_calibration(tmp_path, sample)
    with (tmp_path / 'calibration.csv').open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames[:4] == ['pair_id', 'case_number', 'search_query', 'authority']
    assert reader.fieldnames[-2:] == ['human_grade', 'human_note']
    assert not {'stratum', 'ranks', 'is_gold_chunk'} & set(reader.fieldnames)
    assert rows[0]['human_grade'] == ''
    manifest = json.loads((tmp_path / 'calibration.json').read_text())
    assert len(manifest['pairs']) == 50
    with pytest.raises(FileExistsError):
        write_calibration(tmp_path, sample)


def test_human_grades_reject_values_outside_0_1_2(tmp_path):
    path = tmp_path / 'calibration.csv'
    path.write_text('pair_id,human_grade\nc1|k1,2\nc1|k2,\nc1|k3,5\n', encoding='utf-8-sig')
    with pytest.raises(ValueError, match='c1\\|k3'):
        read_human_grades(path)
    path.write_text('pair_id,human_grade\nc1|k1,2\nc1|k2,\n', encoding='utf-8-sig')
    assert read_human_grades(path) == {('c1', 'k1'): 2}


def test_agreement_on_known_matrix():
    human = {('c', str(i)): g for i, g in enumerate([2, 2, 1, 0, 0, 1])}
    llm = {('c', str(i)): g for i, g in enumerate([2, 1, 1, 0, 0, 2])}
    result = agreement(human, llm, gold_pairs={('c', '0'), ('c', '3')})
    assert result['pairs'] == 6
    assert result['confusion'] == [[2, 0, 0], [0, 1, 1], [0, 1, 1]]   # rows human 0/1/2, cols llm 0/1/2
    assert result['exact_agreement'] == pytest.approx(4 / 6)
    assert result['binary_grade2']['precision'] == pytest.approx(1 / 2)
    assert result['binary_grade2']['recall'] == pytest.approx(1 / 2)
    assert result['binary_grade2']['kappa'] == pytest.approx(0.25)
    assert result['gold_llm_grades'] == {'0': 1, '1': 0, '2': 1}
    assert result['missing_llm'] == 0
```

(Binary kappa check: labels human `[1,1,0,0,0,0]`, llm `[1,0,0,0,0,1]` → p_o = 4/6, p_e = (2·2 + 4·4)/36 = 20/36, κ = (24/36 − 20/36)/(16/36) = 0.25.)

**Step 2: Run to verify failure**

Run: `uv run pytest tests/legal_rag/evaluation/test_calibration.py -v`
Expected: FAIL, module not found

**Step 3: Implement**

```python
"""Stratified human calibration of the relevance judge.

Pairs are drawn before any LLM grade is visible, stratified so the sample is not nearly all
grade 0. The CSV leads with a searchable case number and hides strata, ranks and gold status.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

STRATA = {'gold': 10, 'rank_1_3': 15, 'rank_4_10': 15, 'rank_11_20': 10}
CSV_FIELDS = ['pair_id', 'case_number', 'search_query', 'authority', 'date', 'title', 'document_type',
              'document_id', 'question', 'reference_answer', 'chunk_text', 'human_grade', 'human_note']


def _stratum(row: dict) -> str | None:
    if row['is_gold_chunk']:
        return 'gold'
    best = min(row['ranks'].values(), default=None)
    if best is None:
        return None
    return 'rank_1_3' if best <= 3 else 'rank_4_10' if best <= 10 else 'rank_11_20' if best <= 20 else None


def sample_calibration(pool: list[dict], *, seed: int, quotas: dict[str, int] = STRATA) -> list[dict]:
    rng = random.Random(seed)
    ordered = sorted(pool, key=lambda row: (row['case_id'], row['chunk_id']))
    sample = []
    for name, quota in quotas.items():
        members = [row for row in ordered if _stratum(row) == name]
        if len(members) < quota:
            raise ValueError(f'stratum {name} has {len(members)} pairs, needs {quota}')
        sample += [dict(row, stratum=name) for row in rng.sample(members, quota)]
    return sample


def _pair_id(row: dict) -> str:
    return f"{row['case_id']}|{row['chunk_id']}"


def write_calibration(output: Path, sample: list[dict]) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Shuffle presentation by pair id so strata are not visible from row order.
    rows = sorted(sample, key=_pair_id)
    with (output / 'calibration.json').open('x', encoding='utf-8') as handle:
        json.dump({'pairs': [{'pair_id': _pair_id(r), 'stratum': r['stratum']} for r in rows]}, handle,
                  ensure_ascii=False, indent=2)
    with (output / 'calibration.csv').open('x', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{key: row.get(key) for key in CSV_FIELDS}, 'pair_id': _pair_id(row),
                             'chunk_text': row['text'], 'human_grade': '', 'human_note': ''})
    return output


def read_human_grades(path: Path) -> dict[tuple[str, str], int]:
    grades = {}
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            value = (row.get('human_grade') or '').strip()
            if not value:
                continue
            if value not in {'0', '1', '2'}:
                raise ValueError(f"{row['pair_id']}: human_grade must be 0, 1 or 2, got {value!r}")
            case_id, chunk_id = row['pair_id'].split('|', 1)
            grades[(case_id, chunk_id)] = int(value)
    return grades


def _kappa(a: list[int], b: list[int], weighted: bool) -> float | None:
    labels = sorted(set(a) | set(b) | ({0, 1, 2} if weighted else set()))
    n = len(a)
    if not n:
        return None
    weight = (lambda i, j: abs(i - j) / (len(labels) - 1)) if weighted else (lambda i, j: float(i != j))
    observed = sum(weight(x, y) for x, y in zip(a, b)) / n
    expected = sum(a.count(i) * b.count(j) * weight(i, j) for i in labels for j in labels) / n ** 2
    return None if expected == 0 else 1 - observed / expected


def agreement(human: dict, llm: dict, *, gold_pairs: set) -> dict:
    pairs = sorted(key for key in human if key in llm)
    h, l = [human[k] for k in pairs], [llm[k] for k in pairs]
    confusion = [[sum(x == i and y == j for x, y in zip(h, l)) for j in range(3)] for i in range(3)]
    bh, bl = [int(x == 2) for x in h], [int(y == 2) for y in l]
    tp = sum(x and y for x, y in zip(bh, bl))
    return {
        'pairs': len(pairs), 'missing_llm': len([k for k in human if k not in llm]),
        'confusion': confusion,
        'exact_agreement': sum(x == y for x, y in zip(h, l)) / len(pairs) if pairs else None,
        'weighted_kappa': _kappa(h, l, weighted=True),
        'binary_grade2': {'kappa': _kappa(bh, bl, weighted=False),
                          'precision': tp / sum(bl) if sum(bl) else None,
                          'recall': tp / sum(bh) if sum(bh) else None},
        'gold_llm_grades': {str(g): sum(llm[k] == g for k in pairs if k in gold_pairs) for g in range(3)},
    }
```

**Step 4: Run to verify pass**

Run: `uv run pytest tests/legal_rag/evaluation/test_calibration.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add src/legal_rag/evaluation/calibration.py tests/legal_rag/evaluation/test_calibration.py
git commit -m "feat: stratified judge calibration with searchable case numbers"
```

---

### Task 7: Graded metrics

**Files:**
- Create: `src/legal_rag/evaluation/graded_metrics.py`
- Test: `tests/legal_rag/evaluation/test_graded_metrics.py`

**Step 1: Write the failing test**

```python
import math

import pytest

from legal_rag.evaluation.graded_metrics import case_metrics, summarize_runs


def ranked(*pairs):
    return [{'chunk_id': c, 'document_id': d} for c, d in pairs]


def test_graded_ndcg_success_mrr_and_coverage():
    grades = {'a': 0, 'b': 2, 'c': 1, 'z': 2}               # z judged but not retrieved
    result = case_metrics(ranked(('a', 'A'), ('b', 'B'), ('x', 'X'), ('c', 'C')), grades, level='chunk')
    dcg = 3 / math.log2(3) + 1 / math.log2(5)
    idcg = 3 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)
    assert result['ndcg@10'] == pytest.approx(dcg / idcg)
    assert result['success@5'] == 1 and result['mrr'] == pytest.approx(1 / 2)
    assert result['precision@10_grade2'] == pytest.approx(1 / 10)
    assert result['judged_coverage@10'] == pytest.approx(3 / 4)


def test_document_level_takes_first_chunk_rank_and_max_grade():
    grades = {'a1': 0, 'a2': 2, 'b1': 1, 'c9': 2}
    documents = {'a1': 'A', 'a2': 'A', 'b1': 'B', 'c9': 'C'}   # c9 judged via another run, not retrieved here
    result = case_metrics(ranked(('a1', 'A'), ('b1', 'B')), grades, level='document', documents=documents)
    assert result['mrr'] == 1.0                                # document A = max(0, 2) at rank 1
    assert result['judged_coverage@10'] == 1.0
    ideal = 3 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)   # A=2, C=2, B=1
    assert result['ndcg@10'] == pytest.approx((3 / math.log2(2) + 1 / math.log2(3)) / ideal)


def test_case_without_relevant_chunks_has_no_ndcg():
    result = case_metrics(ranked(('a', 'A')), {'a': 0}, level='chunk')
    assert result['ndcg@10'] is None and result['mrr'] == 0.0


def test_summary_means_skip_undefined_ndcg_and_report_counts():
    runs = {'dense': [{'case_id': 'c1', 'ranked': ranked(('a', 'A'))},
                      {'case_id': 'c2', 'ranked': ranked(('b', 'B'))}]}
    grades = {('c1', 'a'): 2, ('c2', 'b'): 0}
    summary = summarize_runs(runs, grades, {('c1', 'a'): 'A', ('c2', 'b'): 'B'})
    chunk = summary['dense']['chunk']
    assert chunk['cases'] == 2 and chunk['cases_without_relevant'] == 1
    assert chunk['ndcg@10'] == 1.0 and chunk['success@5'] == 0.5
    assert chunk['min_judged_coverage@10'] == pytest.approx(1.0)
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/legal_rag/evaluation/test_graded_metrics.py -v`
Expected: FAIL, module not found

**Step 3: Implement**

```python
"""Graded retrieval metrics over pooled LLM judgments.

Unjudged chunks count as grade 0 in the metrics, and the fraction judged is reported as
``judged_coverage@10``. A comparison is valid only when every run's coverage is 1.0.
The ideal ranking uses every judged chunk for the case, so it does not depend on one run.
"""
from __future__ import annotations

import math
from statistics import mean

SUCCESS_KS = (5, 10, 20)


def _dcg(gains: list[int]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(gains, 1))


def case_metrics(ranked: list[dict], grades: dict[str, int], *, level: str,
                 documents: dict[str, str] | None = None) -> dict:
    """``grades`` maps chunk id to grade for this case; ``documents`` maps judged chunk id to
    document id (from the store, so chunks judged for other runs still count in the ideal)."""
    if level == 'document':
        documents = {**{h['chunk_id']: h['document_id'] for h in ranked}, **(documents or {})}
        labels: dict[str, int] = {}
        for chunk_id, grade in grades.items():
            doc = documents[chunk_id]
            labels[doc] = max(labels.get(doc, 0), grade)
        items = list(dict.fromkeys(hit['document_id'] for hit in ranked))
    else:
        items, labels = [h['chunk_id'] for h in ranked], dict(grades)
    gains = [labels.get(item, 0) for item in items]
    ideal = _dcg(sorted(labels.values(), reverse=True)[:10])
    first = next((i for i, g in enumerate(gains, 1) if g == 2), None)
    top10 = items[:10]
    return {
        'ndcg@10': _dcg(gains[:10]) / ideal if ideal else None,
        **{f'success@{k}': int(any(g == 2 for g in gains[:k])) for k in SUCCESS_KS},
        'mrr': 1 / first if first else 0.0,
        'precision@10_grade1': sum(g >= 1 for g in gains[:10]) / 10,
        'precision@10_grade2': sum(g == 2 for g in gains[:10]) / 10,
        'judged_coverage@10': sum(item in labels for item in top10) / len(top10) if top10 else 1.0,
    }


def summarize_runs(runs: dict[str, list[dict]], grades: dict[tuple[str, str], int],
                   documents: dict[tuple[str, str], str]) -> dict:
    per_case_grades: dict[str, dict[str, int]] = {}
    per_case_docs: dict[str, dict[str, str]] = {}
    for (case_id, chunk_id), grade in grades.items():
        per_case_grades.setdefault(case_id, {})[chunk_id] = grade
        per_case_docs.setdefault(case_id, {})[chunk_id] = documents[(case_id, chunk_id)]
    summary = {}
    for mode, rows in runs.items():
        summary[mode] = {}
        for level in ('chunk', 'document'):
            results = [case_metrics(row['ranked'], per_case_grades.get(row['case_id'], {}), level=level,
                                    documents=per_case_docs.get(row['case_id'], {}))
                       for row in rows]
            ndcgs = [r['ndcg@10'] for r in results if r['ndcg@10'] is not None]
            keys = [k for k in results[0] if k not in ('ndcg@10', 'judged_coverage@10')]
            summary[mode][level] = {
                'cases': len(results), 'cases_without_relevant': len(results) - len(ndcgs),
                'ndcg@10': mean(ndcgs) if ndcgs else None,
                **{k: mean(r[k] for r in results) for k in keys},
                'mean_judged_coverage@10': mean(r['judged_coverage@10'] for r in results),
                'min_judged_coverage@10': min(r['judged_coverage@10'] for r in results),
            }
    return summary
```

Note on `cases_without_relevant`: it counts cases whose judged pool holds no chunk graded ≥ 1 (ideal DCG 0). Their nDCG is undefined and excluded from the mean; Success and MRR still count them as 0.

**Step 4: Run to verify pass**

Run: `uv run pytest tests/legal_rag/evaluation/test_graded_metrics.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add src/legal_rag/evaluation/graded_metrics.py tests/legal_rag/evaluation/test_graded_metrics.py
git commit -m "feat: graded nDCG, success and coverage metrics"
```

---

### Task 8: Command-line scripts

**Files:**
- Create: `scripts/sample_calibration.py`, `scripts/judge_relevance.py`, `scripts/judge_agreement.py`, `scripts/eval_graded.py`

Thin wiring over tested modules; verify each with `--help` and the live steps in Task 9.

**`scripts/sample_calibration.py`**

```python
#!/usr/bin/env python
"""Draw the stratified 50-pair human calibration set from a relevance pool."""
from pathlib import Path

import typer

from legal_rag.evaluation.calibration import sample_calibration, write_calibration
from legal_rag.evaluation.relevance_pool import read_jsonl

app = typer.Typer(add_completion=False)


@app.command()
def main(pool_dir: Path = typer.Option(..., exists=True, file_okay=False), seed: int = typer.Option(42)) -> None:
    sample = sample_calibration(read_jsonl(pool_dir / 'pool.jsonl'), seed=seed)
    write_calibration(pool_dir, sample)
    typer.echo(f'wrote {pool_dir / "calibration.csv"}: {len(sample)} pairs — fill human_grade (0/1/2)')


if __name__ == '__main__':
    app()
```

**`scripts/judge_relevance.py`**

```python
#!/usr/bin/env python
"""Grade pool pairs with an Ollama judge, appending to judgments.jsonl. Safe to interrupt and rerun."""
import json
from pathlib import Path

import typer

from legal_rag.evaluation.judgment_store import JudgmentStore
from legal_rag.evaluation.relevance_judge import judge_pool
from legal_rag.evaluation.relevance_pool import read_jsonl
from legal_rag.llm import OllamaChatModel

app = typer.Typer(add_completion=False)


@app.command()
def main(
    pool_dir: Path = typer.Option(..., exists=True, file_okay=False),
    model: str = typer.Option('qwen3:8b'),
    base_url: str = typer.Option('http://localhost:11434'),
    calibration_only: bool = typer.Option(False, help='Judge only the pairs in calibration.json.'),
    pool_file: Path = typer.Option(None, help='Another pool file, e.g. from a new run; defaults to pool.jsonl.'),
) -> None:
    pairs = read_jsonl(pool_file or pool_dir / 'pool.jsonl')
    if calibration_only:
        wanted = {p['pair_id'] for p in json.loads((pool_dir / 'calibration.json').read_text())['pairs']}
        pairs = [p for p in pairs if f"{p['case_id']}|{p['chunk_id']}" in wanted]
    chat = OllamaChatModel(model=model, base_url=base_url, timeout=300.0, think=False)
    counts = judge_pool(pairs, chat, JudgmentStore(pool_dir / 'judgments.jsonl'),
                        progress=lambda message: typer.echo(message, err=True))
    typer.echo(counts)


if __name__ == '__main__':
    app()
```

**`scripts/judge_agreement.py`**

```python
#!/usr/bin/env python
"""Compare human calibration grades with the judge's grades."""
import json
from pathlib import Path

import typer

from legal_rag.evaluation.calibration import agreement, read_human_grades
from legal_rag.evaluation.judgment_store import JudgeKey, JudgmentStore
from legal_rag.evaluation.relevance_judge import PROMPT_SHA256
from legal_rag.evaluation.relevance_pool import read_jsonl

app = typer.Typer(add_completion=False)


@app.command()
def main(pool_dir: Path = typer.Option(..., exists=True, file_okay=False),
         model: str = typer.Option('qwen3:8b')) -> None:
    human = read_human_grades(pool_dir / 'calibration.csv')
    llm = JudgmentStore(pool_dir / 'judgments.jsonl').grades(JudgeKey(f'ollama:{model}', PROMPT_SHA256))
    gold = {(r['case_id'], r['chunk_id']) for r in read_jsonl(pool_dir / 'pool.jsonl') if r['is_gold_chunk']}
    result = agreement(human, llm, gold_pairs=gold)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    kappa = result['binary_grade2']['kappa']
    typer.echo('PASS: judge usable' if kappa is not None and kappa >= 0.6 else
               'FAIL: binary grade-2 kappa below 0.6 — revise the prompt before the full run')


if __name__ == '__main__':
    app()
```

**`scripts/eval_graded.py`**

```python
#!/usr/bin/env python
"""Graded metrics for every recorded run in a pool directory."""
import json
from pathlib import Path

import typer

from legal_rag.evaluation.graded_metrics import summarize_runs
from legal_rag.evaluation.judgment_store import JudgeKey, JudgmentStore
from legal_rag.evaluation.relevance_judge import PROMPT_SHA256
from legal_rag.evaluation.relevance_pool import read_jsonl

app = typer.Typer(add_completion=False)
ROWS = ['ndcg@10', 'success@5', 'success@10', 'success@20', 'mrr', 'precision@10_grade1',
        'precision@10_grade2', 'min_judged_coverage@10']


@app.command()
def main(pool_dir: Path = typer.Option(..., exists=True, file_okay=False),
         model: str = typer.Option('qwen3:8b'), output: Path = typer.Option(None)) -> None:
    key = JudgeKey(f'ollama:{model}', PROMPT_SHA256)
    store = JudgmentStore(pool_dir / 'judgments.jsonl')
    runs = {path.stem: read_jsonl(path) for path in sorted((pool_dir / 'runs').glob('*.jsonl'))}
    summary = summarize_runs(runs, store.grades(key), store.documents(key))
    summary['_judge'] = {'model': key.model, 'prompt_sha256': key.prompt_sha256,
                         'failed_parses': len(store.failures(key))}
    lines = []
    for level in ('chunk', 'document'):
        lines += [f'\n## {level}\n', '| Metric | ' + ' | '.join(runs) + ' |', '|---|' + '---:|' * len(runs)]
        lines += [f'| {row} | ' + ' | '.join(
            '—' if summary[m][level][row] is None else f'{summary[m][level][row]:.3f}' for m in runs) + ' |'
            for row in ROWS]
    if any(summary[m][l]['min_judged_coverage@10'] < 1 for m in runs for l in ('chunk', 'document')):
        lines.append('\n**WARNING:** some top-10 chunks are unjudged; run judge_relevance.py first.')
    text = '\n'.join(lines)
    typer.echo(text)
    if output:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'graded.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        (output / 'graded.md').write_text(f'# Graded retrieval comparison\n{text}\n', encoding='utf-8')


if __name__ == '__main__':
    app()
```

**Step 1:** create the four files. **Step 2:** `uv run python scripts/<each>.py --help` exits 0. Confirm `from legal_rag.llm import OllamaChatModel` works (`src/legal_rag/llm/__init__.py`); if not exported, import from `legal_rag.llm.ollama`. **Step 3:** `uv run pytest -q` — all pass.

**Step 4: Commit**

```bash
git add scripts/sample_calibration.py scripts/judge_relevance.py scripts/judge_agreement.py scripts/eval_graded.py
git commit -m "feat: scripts for calibration, judging, agreement and graded evaluation"
```

---

### Task 9: Live run up to the human checkpoint

**Step 1:** Build the pool (Task 5 Step 3–4).

**Step 2:** Check chunk lengths against the prompt cap:

```bash
uv run python -c "import json,statistics as s;L=[len(json.loads(l)['text']) for l in open('data/eval/relevance/ip-v1-pilot-200/pool.jsonl')];print(len(L),s.median(L),sorted(L)[int(.95*len(L))],max(L))"
```

If the 95th percentile exceeds `MAX_CHUNK_CHARS` (2500), report it to the user before judging — truncation may hide the relevant passage.

**Step 3:** `uv run python scripts/sample_calibration.py --pool-dir data/eval/relevance/ip-v1-pilot-200`

**Step 4: STOP.** Hand `calibration.csv` to the user for labelling. Do not run the judge on these pairs, and do not show LLM grades, until the human grades are filled in.

**Step 5 (after the human labels):**

```bash
uv run python scripts/judge_relevance.py --pool-dir data/eval/relevance/ip-v1-pilot-200 --calibration-only
uv run python scripts/judge_agreement.py --pool-dir data/eval/relevance/ip-v1-pilot-200
```

On PASS, the full judge run (`judge_relevance.py` without `--calibration-only`) goes in the background, then `eval_graded.py --output …`. On FAIL, revise the prompt in `relevance_judge.py` (which changes `PROMPT_SHA256`) and repeat Step 5.

---

### Task 10: Documentation

**Files:**
- Modify: `docs/retrieval-diagnostics.md` — append a "Graded relevance evaluation" section.

Content: why exact-gold is unusable (0 of 76,142 questions name a document), the directory layout above, the five commands in order, the human checkpoint, the kappa ≥ 0.6 gate, what `judged_coverage@10 < 1` means and how to top up with `--pool-file` for a new run, and the ip-v1 + domain-filter scope (fan-out is not measured).

```bash
git add docs/retrieval-diagnostics.md
git commit -m "docs: graded relevance evaluation workflow"
```

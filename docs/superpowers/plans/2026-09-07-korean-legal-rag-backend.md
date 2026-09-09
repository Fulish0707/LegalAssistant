# Korean Legal RAG Backend Implementation Plan (V0.1–V0.3)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the `legal_rag` backend from raw AI Hub-style files to a verified, source-cited answer over a CLI and a FastAPI endpoint, exactly as specified in `docs/superpowers/specs/2026-09-07-korean-legal-rag-backend-design.md`.

**Architecture:** Plain Python stages behind `typing.Protocol` interfaces, composed by one `Orchestrator`. Offline path: ingestion → chunking → BM25 statistics + dense embeddings → embedded Qdrant. Online path: analyzer → clarification policy → rewriter → hybrid retrieval with RRF → reranker → evidence builder → structured LLM answer → deterministic citation verification → policy. Everything is configured through YAML profiles plus environment variables, and every stage has a fake backend so the full pipeline is testable offline.

**Tech Stack:** Python 3.11 via `uv`, Pydantic v2, pydantic-settings, PyYAML, qdrant-client (embedded mode), httpx (Ollama), Typer, FastAPI, pytest. Optional extras: kiwipiepy, sentence-transformers.

---

## Conventions for every task

- Work in the worktree `/home/layke/Documents/MyLLMAgent/.claude/worktrees/file-responsibilities-871bad`.
- Run tests with `uv run pytest tests/legal_rag -q`. Run a single test with `uv run pytest <path>::<name> -v`.
- Commit after each task with the message given. End commit messages with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Every schema uses `model_config = ConfigDict(frozen=True, extra="forbid")` unless the task says otherwise.
- Never import `qdrant_client` outside `src/legal_rag/retrieval/store.py`, never import `httpx` outside `embeddings/ollama.py` and `llm/ollama.py`.
- Deviation from the design, recorded here so both stay consistent: the reference regexes (statute articles, case numbers) live in the leaf module `src/legal_rag/textutil/refs.py` so that `ingestion` can use them without depending on `verification`. `verification/citations.py` wraps them. A composition root `src/legal_rag/factory.py` builds backends from `Settings` for both scripts and the API.

---

## V0.1 — Data foundation

### Task 1: Project skeleton and tooling

**Files:**
- Create: `pyproject.toml`
- Create: `src/legal_rag/__init__.py`
- Create: `src/legal_rag/textutil/__init__.py`, and empty `__init__.py` in `config`, `schemas`, `ingestion`, `ingestion/normalize`, `chunking`, `embeddings`, `retrieval`, `reranking`, `llm`, `rag`, `verification`, `evaluation`, `api`
- Create: `tests/legal_rag/__init__.py`, `tests/legal_rag/unit/__init__.py`, `tests/legal_rag/integration/__init__.py`, `tests/legal_rag/live/__init__.py`, `tests/legal_rag/fixtures/.gitkeep`
- Create: `data/raw/.gitkeep`, `data/processed/.gitkeep`, `data/indexes/.gitkeep`, `data/eval/.gitkeep`
- Modify: `.gitignore` (append `data/**` ignore with `.gitkeep` exceptions, `.venv/`, `uv.lock` stays tracked)

**Step 1: Write `pyproject.toml`**

```toml
[project]
name = "legal-rag"
version = "0.1.0"
description = "Korean legal RAG assistant backend"
requires-python = ">=3.11,<3.12"
dependencies = [
  "pydantic>=2.7",
  "pydantic-settings>=2.3",
  "pyyaml>=6.0",
  "qdrant-client>=1.10",
  "httpx>=0.27",
  "typer>=0.12",
  "fastapi>=0.111",
  "uvicorn>=0.30",
]

[project.optional-dependencies]
kiwi = ["kiwipiepy>=0.17"]
rerank = ["sentence-transformers>=3.0"]

[dependency-groups]
dev = ["pytest>=8.2", "pytest-cov>=5.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/legal_rag"]

[tool.pytest.ini_options]
testpaths = ["tests/legal_rag"]
addopts = "-p no:cacheprovider"
```

**Step 2: Provision the environment**

Run:
```bash
uv python install 3.11
uv sync --group dev
```
Expected: `.venv` created, `uv.lock` written.

**Step 3: Write `src/legal_rag/__init__.py`**

```python
"""Korean legal RAG backend."""

__version__ = "0.1.0"
```

**Step 4: Write a smoke test `tests/legal_rag/unit/test_package.py`**

```python
import legal_rag


def test_package_imports():
    assert legal_rag.__version__ == "0.1.0"
```

**Step 5: Run and verify**

Run: `uv run pytest tests/legal_rag -q`
Expected: `1 passed`

**Step 6: Append to `.gitignore`**

```
# legal_rag data and env
.venv/
data/**
!data/**/
!data/**/.gitkeep
```

**Step 7: Commit**

```bash
git add pyproject.toml uv.lock src tests/legal_rag data .gitignore
git commit -m "chore: scaffold the legal_rag package with uv and pytest"
```

---

### Task 2: Settings and YAML profiles

**Files:**
- Create: `src/legal_rag/config/settings.py`, `src/legal_rag/config/loader.py`
- Create: `configs/default.yaml`, `configs/local.yaml`, `configs/colab.yaml`
- Test: `tests/legal_rag/unit/test_settings.py`

**Step 1: Write the failing tests**

```python
import os
from pathlib import Path

import pytest

from legal_rag.config import Settings, load_settings


def test_default_profile_loads_with_expected_defaults():
    settings = load_settings(profile="default")
    assert settings.embedding.backend == "ollama"
    assert settings.embedding.model == "bge-m3"
    assert settings.retrieval.k_final == 5
    assert settings.qdrant.mode == "embedded"


def test_profile_overrides_default(tmp_path: Path):
    (tmp_path / "default.yaml").write_text("retrieval:\n  k_final: 5\n", encoding="utf-8")
    (tmp_path / "fast.yaml").write_text("retrieval:\n  k_final: 3\n", encoding="utf-8")
    settings = load_settings(profile="fast", config_dir=tmp_path)
    assert settings.retrieval.k_final == 3


def test_environment_overrides_yaml(monkeypatch, tmp_path: Path):
    (tmp_path / "default.yaml").write_text("retrieval:\n  k_final: 5\n", encoding="utf-8")
    monkeypatch.setenv("LEGAL_RAG__RETRIEVAL__K_FINAL", "7")
    settings = load_settings(profile="default", config_dir=tmp_path)
    assert settings.retrieval.k_final == 7


def test_secrets_in_yaml_are_rejected(tmp_path: Path):
    (tmp_path / "default.yaml").write_text("qdrant:\n  api_key: leaked\n", encoding="utf-8")
    with pytest.raises(ValueError, match="api_key"):
        load_settings(profile="default", config_dir=tmp_path)


def test_hash_is_stable_and_ignores_secrets(monkeypatch):
    a = load_settings(profile="default").hash()
    monkeypatch.setenv("LEGAL_RAG__QDRANT__API_KEY", "secret")
    b = load_settings(profile="default").hash()
    assert a == b
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/legal_rag/unit/test_settings.py -q`
Expected: `ImportError` on `legal_rag.config`.

**Step 3: Implement `settings.py`**

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsSettings(_Section):
    root: Path = Path(".")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    index_dir: Path = Path("data/indexes")
    eval_dir: Path = Path("data/eval")


class CorpusSettings(_Section):
    version: str = "v1"


class EmbeddingSettings(_Section):
    backend: Literal["ollama", "fake"] = "ollama"
    model: str = "bge-m3"
    base_url: str = "http://localhost:11434"
    batch_size: int = 16
    fake_dimension: int = 64


class SparseSettings(_Section):
    tokenizer: Literal["regex", "kiwi"] = "regex"
    k1: float = 1.2
    b: float = 0.75


class QdrantSettings(_Section):
    mode: Literal["embedded", "server"] = "embedded"
    path: Path | None = None            # defaults to <index_dir>/<corpus_version>/qdrant
    url: str = "http://localhost:6333"
    api_key: SecretStr | None = None    # environment only


class RetrievalSettings(_Section):
    k_dense: int = 20
    k_sparse: int = 20
    k_fused: int = 20
    k_final: int = 5
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0


class RerankSettings(_Section):
    backend: Literal["noop", "cross_encoder"] = "noop"
    model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cpu"


class LLMSettings(_Section):
    backend: Literal["ollama", "fake", "transformers"] = "ollama"
    model: str = "qwen3:4b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 1024
    adapter: str | None = None
    timeout_seconds: float = 120.0


class RagSettings(_Section):
    allow_clarification: bool = True
    llm_rewrite: bool = False
    history_turns: int = 4
    max_regenerations: int = 1
    max_evidence_chars: int = 1500


class ApiSettings(_Section):
    host: str = "127.0.0.1"
    port: int = 8000


class TraceSettings(_Section):
    include_text: bool = False
    path: Path | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LEGAL_RAG__",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    paths: PathsSettings = Field(default_factory=PathsSettings)
    corpus: CorpusSettings = Field(default_factory=CorpusSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    sparse: SparseSettings = Field(default_factory=SparseSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    rag: RagSettings = Field(default_factory=RagSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    trace: TraceSettings = Field(default_factory=TraceSettings)

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        # Environment beats YAML (passed as init kwargs), which beats defaults.
        return (env_settings, init_settings)

    # Derived paths -------------------------------------------------------
    def processed_corpus_dir(self) -> Path:
        return self.paths.root / self.paths.processed_dir / self.corpus.version

    def index_corpus_dir(self) -> Path:
        return self.paths.root / self.paths.index_dir / self.corpus.version

    def qdrant_path(self) -> Path:
        return self.qdrant.path or (self.index_corpus_dir() / "qdrant")

    def hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"qdrant": {"api_key"}})
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
```

**Step 4: Implement `loader.py`**

```python
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .settings import Settings

SECRET_PATHS = (("qdrant", "api_key"),)
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return data


def _reject_secrets(data: dict[str, Any], path: Path) -> None:
    for section, key in SECRET_PATHS:
        if isinstance(data.get(section), dict) and key in data[section]:
            raise ValueError(f"{path}: {section}.{key} must come from the environment, not YAML")


def load_settings(profile: str | None = None, config_dir: Path | None = None, overrides: dict[str, Any] | None = None) -> Settings:
    config_dir = config_dir or DEFAULT_CONFIG_DIR
    profile = profile or os.getenv("LEGAL_RAG_PROFILE", "local")
    merged: dict[str, Any] = {}
    for name in ("default", profile):
        path = config_dir / f"{name}.yaml"
        data = _read_yaml(path)
        _reject_secrets(data, path)
        merged = _deep_merge(merged, data)
    if overrides:
        merged = _deep_merge(merged, overrides)
    return Settings(**merged)
```

`config/__init__.py` exports `Settings`, `load_settings`.

**Step 5: Write the YAML profiles**

`configs/default.yaml` lists every section with the defaults above (copy the values, with comments). `configs/local.yaml`:

```yaml
# Ubuntu host with RTX 4060. Ollama serves both the LLM and the embedder.
embedding:
  model: bge-m3
llm:
  model: qwen3:4b
```

`configs/colab.yaml`:

```yaml
# Google Colab. Mount Drive and point paths at it; Qdrant stays embedded.
paths:
  root: /content/drive/MyDrive/legal_rag
embedding:
  batch_size: 64
```

**Step 6: Run tests**

Run: `uv run pytest tests/legal_rag/unit/test_settings.py -q`
Expected: `5 passed`

**Step 7: Commit**

```bash
git add src/legal_rag/config configs tests/legal_rag/unit/test_settings.py
git commit -m "feat: add layered settings with YAML profiles and env overrides"
```

---

### Task 3: Reference extraction utilities (`textutil`)

**Files:**
- Create: `src/legal_rag/textutil/refs.py`, `src/legal_rag/textutil/dates.py`, `src/legal_rag/textutil/sentences.py`
- Test: `tests/legal_rag/unit/test_textutil.py`

**Step 1: Failing tests**

```python
from datetime import date

from legal_rag.textutil.dates import parse_korean_date
from legal_rag.textutil.refs import extract_case_refs, extract_statute_refs, normalize_article
from legal_rag.textutil.sentences import split_sentences


def test_statute_refs_with_law_name_and_sub_article():
    text = "주택임대차보호법 제3조의2 제2항과 민법 제750조에 따라"
    refs = extract_statute_refs(text)
    assert [r.render() for r in refs] == ["주택임대차보호법 제3조의2", "민법 제750조"]
    assert refs[0].paragraph == "제2항"


def test_bare_article_without_law_name():
    refs = extract_statute_refs("같은 법 제10조에 의하면")
    assert refs[0].law_name is None and refs[0].article == "제10조"


def test_case_refs_ignore_dates():
    text = "대법원 2019. 5. 30. 선고 2018다12345 판결, 2020도1234, 2021년12월"
    assert extract_case_refs(text) == ["2018다12345", "2020도1234"]


def test_normalize_article_spacing():
    assert normalize_article("제 3 조의 2") == "제3조의2"


def test_parse_dates():
    assert parse_korean_date("2019.01.02") == date(2019, 1, 2)
    assert parse_korean_date("20190102") == date(2019, 1, 2)
    assert parse_korean_date("2019. 1. 2.") == date(2019, 1, 2)
    assert parse_korean_date("2019-01-02") == date(2019, 1, 2)
    assert parse_korean_date("") is None


def test_split_sentences_korean():
    text = "임차인은 보증금을 돌려받을 수 있다. 다만 등기명령을 먼저 신청해야 한다! 정말인가?"
    assert split_sentences(text) == [
        "임차인은 보증금을 돌려받을 수 있다.",
        "다만 등기명령을 먼저 신청해야 한다!",
        "정말인가?",
    ]
```

**Step 2: Run, expect ImportError.**

**Step 3: Implement `refs.py`**

```python
from __future__ import annotations

import re
from dataclasses import dataclass

LAW_SUFFIX = r"(?:법|령|규칙|조례|규정)"
_LAW_NAME = rf"(?P<law>[가-힣A-Za-z0-9·]+?{LAW_SUFFIX})"
_ARTICLE = r"제\s*(?P<art>\d+)\s*조(?:\s*의\s*(?P<sub>\d+))?"
_PARAGRAPH = r"(?:\s*제\s*(?P<para>\d+)\s*항)?"
STATUTE_WITH_LAW = re.compile(rf"{_LAW_NAME}\s*{_ARTICLE}{_PARAGRAPH}")
STATUTE_BARE = re.compile(rf"{_ARTICLE}{_PARAGRAPH}")

CASE_TYPES = (
    "가합|가단|가소|고합|고단|고정|다카|"
    "다|도|두|누|나|라|마|노|오|모|르|므|드|느|브|스|수|무|어|우|주|구|카|타|허|후|헌|초|즈|재|하|바"
)
CASE_RE = re.compile(rf"(?<![0-9])(\d{{2,4}}(?:{CASE_TYPES})\d{{1,7}})(?![0-9])")


@dataclass(frozen=True)
class StatuteRef:
    law_name: str | None
    article: str          # "제750조", "제3조의2"
    paragraph: str | None # "제2항"

    def render(self) -> str:
        return f"{self.law_name} {self.article}" if self.law_name else self.article


def normalize_article(raw: str) -> str:
    m = STATUTE_BARE.search(raw)
    if not m:
        return re.sub(r"\s+", "", raw)
    art = f"제{m.group('art')}조"
    if m.group("sub"):
        art += f"의{m.group('sub')}"
    return art


def extract_statute_refs(text: str) -> list[StatuteRef]:
    refs: list[StatuteRef] = []
    consumed: list[tuple[int, int]] = []
    for m in STATUTE_WITH_LAW.finditer(text):
        refs.append(StatuteRef(m.group("law"), _article_from(m), _para_from(m)))
        consumed.append(m.span())
    for m in STATUTE_BARE.finditer(text):
        if any(start <= m.start() < end for start, end in consumed):
            continue
        refs.append(StatuteRef(None, _article_from(m), _para_from(m)))
    refs.sort(key=lambda r: text.find(r.article))
    return _dedupe(refs)


def _article_from(m: re.Match) -> str:
    art = f"제{m.group('art')}조"
    return art + (f"의{m.group('sub')}" if m.group("sub") else "")


def _para_from(m: re.Match) -> str | None:
    return f"제{m.group('para')}항" if m.group("para") else None


def _dedupe(refs: list[StatuteRef]) -> list[StatuteRef]:
    seen: set[tuple] = set()
    out = []
    for r in refs:
        key = (r.law_name, r.article, r.paragraph)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def extract_case_refs(text: str) -> list[str]:
    seen: list[str] = []
    for m in CASE_RE.finditer(text):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return seen
```

Note on ordering in `extract_statute_refs`: sort by first occurrence of the full match, not `text.find(article)`; keep the match start in a tuple while collecting and sort on it. Adjust the code accordingly.

**Step 4: Implement `dates.py`**

```python
from __future__ import annotations

import re
from datetime import date

_PATTERNS = (
    re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$"),
    re.compile(r"^(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.?$"),
    re.compile(r"^(\d{4})(\d{2})(\d{2})$"),
    re.compile(r"^(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일$"),
)


def parse_korean_date(value: str | None) -> date | None:
    if not value:
        return None
    text = value.strip()
    for pattern in _PATTERNS:
        m = pattern.match(text)
        if m:
            year, month, day = (int(g) for g in m.groups())
            try:
                return date(year, month, day)
            except ValueError:
                return None
    return None
```

**Step 5: Implement `sentences.py`**

```python
import re

_SPLIT = re.compile(r"(?<=[.!?。])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SPLIT.split(text.strip()) if s.strip()]
```

**Step 6: Run tests, expect 6 passed. Commit**

```bash
git add src/legal_rag/textutil tests/legal_rag/unit/test_textutil.py
git commit -m "feat: add statute and case reference extraction utilities"
```

---

### Task 4: Canonical document schemas

**Files:**
- Create: `src/legal_rag/schemas/documents.py`, `src/legal_rag/schemas/__init__.py`
- Test: `tests/legal_rag/unit/test_schemas_documents.py`

**Step 1: Failing tests**

```python
import json
from datetime import date

import pytest
from pydantic import TypeAdapter, ValidationError

from legal_rag.schemas import (
    AdminInterpretation, CourtDecision, DecisionSection, DocumentType, LegalDocument,
    SourceRef, Statute, StatuteArticle, StatuteParagraph, make_document_id,
)


def _source():
    return SourceRef(dataset="fixture", file="a.json", record_key="k1")


def test_document_id_is_deterministic():
    a = make_document_id(DocumentType.statute, "aihub", "민법@2020")
    b = make_document_id(DocumentType.statute, "aihub", "민법@2020")
    assert a == b and a.startswith("statute:") and len(a) == len("statute:") + 16


def test_statute_round_trips_through_discriminated_union():
    statute = Statute(
        id=make_document_id(DocumentType.statute, "fixture", "민법"),
        title="민법", text="제750조 ...", source=_source(), law_name="민법",
        version=None, promulgation_date=None,
        articles=[StatuteArticle(number="제750조", title="불법행위의 내용", text="고의 또는 과실로...", paragraphs=[])],
    )
    dumped = statute.model_dump_json()
    parsed = TypeAdapter(LegalDocument).validate_json(dumped)
    assert isinstance(parsed, Statute) and parsed.articles[0].number == "제750조"


def test_court_decision_requires_known_sections():
    with pytest.raises(ValidationError):
        CourtDecision(
            id="court_decision:x", title="t", text="t", source=_source(), court="대법원",
            case_number="2018다12345", decision_date=date(2019, 5, 30), case_type=None,
            sections={"bogus": "..."},
        )


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        AdminInterpretation(
            id="admin_interpretation:x", title="t", text="t", source=_source(), authority="법제처",
            reference_number="19-0001", date=None, subject="s", question="q", interpretation="i", surprise=1,
        )
```

**Step 2: Run, expect ImportError.**

**Step 3: Implement `documents.py` exactly as section 5.1 of the design**, plus:

```python
def make_document_id(document_type: DocumentType, dataset: str, record_key: str) -> str:
    digest = hashlib.sha256(f"{dataset}|{record_key}".encode("utf-8")).hexdigest()[:16]
    return f"{document_type.value}:{digest}"
```

`document_type` on each subclass is `Literal[...]` with a default so callers need not pass it. `sections: dict[DecisionSection, str]` makes unknown keys fail validation. `schemas/__init__.py` re-exports everything.

**Step 4: Run tests, expect 4 passed. Commit**

```bash
git add src/legal_rag/schemas tests/legal_rag/unit/test_schemas_documents.py
git commit -m "feat: add canonical legal document schemas"
```

---

### Task 5: Chunk, query, evidence, answer, verification, evaluation, conversation schemas

**Files:**
- Create: `src/legal_rag/schemas/chunks.py`, `query.py`, `evidence.py`, `answer.py`, `verification.py`, `evaluation.py`, `conversation.py`, `response.py`
- Modify: `src/legal_rag/schemas/__init__.py`
- Test: `tests/legal_rag/unit/test_schemas_misc.py`

**Step 1: Failing tests**

```python
from legal_rag.schemas import (
    Chunk, ChunkMetadata, Citation, CitationKind, Confidence, DocumentType, Evidence,
    EvidenceBundle, LegalAnswer, QueryAnalysis, RetrievalExample, RetrievalFilters, ScoredChunk,
)


def _chunk(cid="statute:abc#art:제750조"):
    return Chunk(
        id=cid, document_id="statute:abc", document_type=DocumentType.statute, path="art:제750조", order=0,
        text="민법 제750조 불법행위의 내용\n고의 또는 과실로 인한 위법행위로 타인에게 손해를 가한 자는 그 손해를 배상할 책임이 있다.",
        title="민법", metadata=ChunkMetadata(law_name="민법", article="제750조", dataset="fixture", corpus_version="v1"),
    )


def test_chunk_metadata_defaults_are_none():
    m = _chunk().metadata
    assert m.court is None and m.paragraph is None and m.effective_date is None


def test_scored_chunk_carries_retriever_name():
    s = ScoredChunk(chunk=_chunk(), score=0.5, retriever="dense")
    assert s.retriever == "dense"


def test_evidence_bundle_renders_source_blocks():
    bundle = EvidenceBundle(items=[
        Evidence(source_id="S1", chunk_id=_chunk().id, document_id="statute:abc", document_type=DocumentType.statute,
                 header={"Type": "Statute", "Law": "민법", "Article": "제750조"}, text="고의 또는 과실로...", score=0.9)
    ])
    rendered = bundle.render()
    assert rendered.startswith("[LEGAL SOURCE S1]")
    assert "Law: 민법" in rendered and "Article: 제750조" in rendered and "Source ID: S1" in rendered


def test_legal_answer_json_schema_has_required_fields():
    schema = LegalAnswer.model_json_schema()
    for field in ("summary", "legal_issues", "analysis", "citations", "confidence"):
        assert field in schema["required"]


def test_query_analysis_defaults():
    qa = QueryAnalysis()
    assert qa.needs_more_info is False and qa.keywords == [] and qa.document_types_needed == []


def test_retrieval_example_parses_filters():
    ex = RetrievalExample(id="q1", query="보증금", relevant_document_ids=["statute:abc"],
                          filters=RetrievalFilters(document_types=[DocumentType.statute]))
    assert ex.filters.document_types == [DocumentType.statute]
```

**Step 2: Run, expect ImportError.**

**Step 3: Implement** each file as in design sections 5.2–5.8. Specifics:

- `ChunkMetadata`: every field optional with `None` default except `dataset: str`, `corpus_version: str`. Add `section: str | None`, `decision_date_int: int | None` and `effective_date_int: int | None` computed by a `model_validator(mode="after")`? No: keep the schema pure and compute the integer forms in `retrieval/store.py` when building payloads.
- `QueryAnalysis`: all fields default to empty/None/False so the LLM can omit them.
- `EvidenceBundle.render()`:

```python
def render(self) -> str:
    blocks = []
    for e in self.items:
        lines = [f"[LEGAL SOURCE {e.source_id}]", ""]
        lines += [f"{k}: {v}" for k, v in e.header.items()]
        lines += [f"Source ID: {e.source_id}", "", "Text:", e.text, ""]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
```

- `response.py`:

```python
class ChatResponse(BaseModel):
    kind: Literal["answer", "clarification", "rejected"]
    conversation_id: str
    answer: LegalAnswer | None = None
    questions: list[str] = []
    reason: str | None = None
    evidence: EvidenceBundle | None = None
    verification: VerificationReport | None = None
    trace: Trace | None = None
```

with `Trace` defined in `schemas/trace.py`:

```python
class StageTiming(BaseModel):
    stage: str
    ms: float

class Trace(BaseModel):
    model_config = ConfigDict(extra="forbid")   # not frozen: built incrementally
    conversation_id: str
    turn_id: str
    model: str | None = None
    timings: list[StageTiming] = []
    analysis: QueryAnalysis | None = None
    search_queries: list[str] = []
    retrieved: dict[str, list[tuple[str, float]]] = {}   # leg name -> [(chunk_id, score)]
    evidence_ids: list[str] = []
    verification: VerificationReport | None = None
    policy_action: str | None = None
    regenerations: int = 0
    user_text: str | None = None
```

**Step 4: Run tests, expect 6 passed. Commit**

```bash
git add src/legal_rag/schemas tests/legal_rag/unit/test_schemas_misc.py
git commit -m "feat: add chunk, query, evidence, answer, verification, and trace schemas"
```

---

### Task 6: Fixture files

**Files:**
- Create: `tests/legal_rag/fixtures/raw/aihub_judgment/judgment_001.json`, `judgment_002.json`
- Create: `tests/legal_rag/fixtures/raw/statutes/housing_lease_act.json`, `civil_act_excerpt.json`
- Create: `tests/legal_rag/fixtures/raw/admin_interpretations/interp_001.json`
- Create: `tests/legal_rag/fixtures/raw/unknown/notes.json`
- Create: `tests/legal_rag/fixtures/eval/retrieval.jsonl`

**Content requirements** (write realistic Korean text, each field a few sentences):

`judgment_001.json` (AI Hub judgment layout, values as lists of strings):
```json
{
  "info": {"caseNm": "보증금반환", "caseNo": "2019가단12345", "courtNm": "서울중앙지방법원", "judmnAdjuDe": "2020.03.15", "caseField": "1", "detailField": "민사"},
  "mentionedItems": {"rqestObjet": ["피고는 원고에게 50,000,000원을 지급하라."]},
  "assrs": {"acusrAssrs": ["원고는 임대차계약이 종료되었으므로 보증금 반환을 구한다."], "dedatAssrs": ["피고는 새 임차인이 구해지지 않아 반환할 수 없다고 주장한다."]},
  "facts": {"bsisFacts": ["원고는 2017. 3. 1. 피고와 임대차계약을 체결하고 보증금 5,000만 원을 지급하였다.", "계약은 2019. 2. 28. 기간 만료로 종료되었다."]},
  "dcss": {"courtDcss": ["주택임대차보호법 제3조의3에 따라 임차인은 임차권등기명령을 신청할 수 있다.", "임대차가 종료되면 임대인은 보증금을 반환할 의무가 있고 새 임차인 확보는 반환의 조건이 아니다."]},
  "close": {"cnclsns": ["피고는 원고에게 50,000,000원 및 지연손해금을 지급하라."]}
}
```

`judgment_002.json`: a 대법원 tort case, `caseNo` `2018다12345`, `judmnAdjuDe` `2019.05.30`, reasoning citing `민법 제750조`, holding about 손해배상. Make `dcss.courtDcss` long enough (≥ 8 sentences, ≥ 1300 characters) so the decision chunker splits it into two windows.

`housing_lease_act.json` (statute_json_v1):
```json
{
  "law_name": "주택임대차보호법",
  "version": "법률 제17363호",
  "promulgation_date": "2020-06-09",
  "effective_date": "2020-12-10",
  "articles": [
    {"number": "제3조", "title": "대항력 등", "chapter": "제1장 총칙", "text": "", "paragraphs": [
      {"number": "제1항", "text": "임대차는 그 등기가 없는 경우에도 임차인이 주택의 인도와 주민등록을 마친 때에는 그 다음 날부터 제삼자에 대하여 효력이 생긴다.", "items": []},
      {"number": "제2항", "text": "...", "items": []}
    ]},
    {"number": "제3조의2", "title": "보증금의 회수", "chapter": "제1장 총칙", "text": "", "paragraphs": [ ... 2 paragraphs ... ]},
    {"number": "제3조의3", "title": "임차권등기명령", "chapter": "제1장 총칙", "text": "", "paragraphs": [
      {"number": "제1항", "text": "임대차가 끝난 후 보증금이 반환되지 아니한 경우 임차인은 ... 임차권등기명령을 신청할 수 있다.", "items": []},
      {"number": "제2항", "text": "임차권등기명령의 신청서에는 다음 각 호의 사항을 적어야 한다.", "items": [
        {"number": "제1호", "text": "신청의 취지 및 이유"}, {"number": "제2호", "text": "임대차의 목적인 주택의 표시"}, {"number": "제3호", "text": "임차권등기의 원인이 된 사실"}]}
    ]}
  ]
}
```
Make `제3조의3` total length exceed 800 characters so it splits per paragraph, and keep `제3조` under 800 so it stays one chunk.

`civil_act_excerpt.json`: 민법 with `제750조` (불법행위의 내용) and `제751조`, short, no paragraphs.

`interp_001.json`: `{"authority": "법제처", "reference_number": "19-0123", "date": "2019-08-01", "subject": "임차권등기명령 신청 요건", "question": "...", "interpretation": "...", "legal_basis": ["주택임대차보호법 제3조의3"]}`.

`unknown/notes.json`: `{"note": "not a legal record"}`.

`retrieval.jsonl` (three lines, `relevant_document_ids` computed with `make_document_id`; write them after Task 8 prints ids, or compute in the test that uses them):
```json
{"id": "q1", "query": "집주인이 보증금을 돌려주지 않습니다", "relevant_document_ids": ["<housing_lease_act id>", "<judgment_001 id>"]}
{"id": "q2", "query": "민법 제750조 불법행위 손해배상", "relevant_document_ids": ["<civil_act id>", "<judgment_002 id>"]}
{"id": "q3", "query": "임차권등기명령 신청서에 적어야 하는 사항", "relevant_document_ids": ["<housing_lease_act id>"], "filters": {"document_types": ["statute"]}}
```
Since the ids are deterministic (`make_document_id(type, dataset, record_key)` with `dataset` = folder name and record_key = case number / law name@version / reference number), compute them once with a one-off `uv run python -c` and paste.

**Commit**

```bash
git add tests/legal_rag/fixtures
git commit -m "test: add AI Hub-style legal fixtures and a retrieval eval sample"
```

---

### Task 7: Raw loaders and schema detection

**Files:**
- Create: `src/legal_rag/ingestion/loaders.py`, `src/legal_rag/ingestion/detect.py`
- Test: `tests/legal_rag/unit/test_ingestion_loaders.py`

**Step 1: Failing tests**

```python
from pathlib import Path

from legal_rag.ingestion.detect import detect_schema
from legal_rag.ingestion.loaders import discover_raw_records

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "raw"


def test_discover_yields_one_record_per_json_object():
    records = list(discover_raw_records(FIXTURES))
    files = {r.file for r in records}
    assert "aihub_judgment/judgment_001.json" in files
    assert all(len(r.file_sha256) == 64 for r in records)
    assert {r.dataset for r in records} == {"aihub_judgment", "statutes", "admin_interpretations", "unknown"}


def test_jsonl_and_list_files_yield_multiple_records(tmp_path: Path):
    (tmp_path / "ds").mkdir()
    (tmp_path / "ds" / "a.jsonl").write_text('{"x":1}\n{"x":2}\n', encoding="utf-8")
    (tmp_path / "ds" / "b.json").write_text('[{"x":3},{"x":4}]', encoding="utf-8")
    records = list(discover_raw_records(tmp_path))
    assert [(r.file, r.index) for r in records] == [("ds/a.jsonl", 0), ("ds/a.jsonl", 1), ("ds/b.json", 0), ("ds/b.json", 1)]


def test_detect_schema_by_fingerprint():
    assert detect_schema({"info": {"caseNo": "x"}, "facts": {}, "dcss": {}, "close": {}}) == "aihub_judgment_v1"
    assert detect_schema({"law_name": "민법", "articles": []}) == "statute_json_v1"
    assert detect_schema({"authority": "a", "reference_number": "1", "question": "q", "interpretation": "i"}) == "admin_interpretation_v1"
    assert detect_schema({"note": "x"}) is None
```

**Step 2: Run, expect ImportError.**

**Step 3: Implement `loaders.py`**

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class RawRecord:
    dataset: str
    file: str          # path relative to raw_dir, POSIX separators
    index: int         # position within the file
    payload: dict[str, Any]
    file_sha256: str


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def discover_raw_records(raw_dir: Path) -> Iterator[RawRecord]:
    raw_dir = Path(raw_dir)
    for path in sorted(p for p in raw_dir.rglob("*") if p.suffix in {".json", ".jsonl"} and p.is_file()):
        rel = path.relative_to(raw_dir).as_posix()
        dataset = rel.split("/", 1)[0] if "/" in rel else "root"
        digest = _sha256(path)
        for index, payload in enumerate(_iter_payloads(path)):
            if isinstance(payload, dict):
                yield RawRecord(dataset, rel, index, payload, digest)


def _iter_payloads(path: Path) -> Iterator[Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        for line in text.splitlines():
            if line.strip():
                yield json.loads(line)
        return
    data = json.loads(text)
    if isinstance(data, list):
        yield from data
    else:
        yield data
```

**Step 4: Implement `detect.py`**

```python
from __future__ import annotations

from typing import Any

SchemaName = str

_FINGERPRINTS: list[tuple[SchemaName, tuple[str, ...]]] = [
    ("aihub_judgment_v1", ("info", "facts", "dcss", "close")),
    ("statute_json_v1", ("law_name", "articles")),
    ("admin_interpretation_v1", ("authority", "reference_number", "question", "interpretation")),
]


def detect_schema(payload: dict[str, Any]) -> SchemaName | None:
    keys = set(payload)
    for name, required in _FINGERPRINTS:
        if all(k in keys for k in required):
            return name
    return None
```

**Step 5: Run, expect 3 passed. Commit**

```bash
git add src/legal_rag/ingestion tests/legal_rag/unit/test_ingestion_loaders.py
git commit -m "feat: discover raw legal records and detect their layout"
```

---

### Task 8: Normalizers for the three layouts

**Files:**
- Create: `src/legal_rag/ingestion/normalize/__init__.py`, `base.py`, `aihub_judgment.py`, `statute_json.py`, `admin_interpretation.py`
- Test: `tests/legal_rag/unit/test_ingestion_normalize.py`

**Step 1: Failing tests**

```python
import json
from datetime import date
from pathlib import Path

import pytest

from legal_rag.ingestion.normalize import NormalizationError, normalize
from legal_rag.schemas import AdminInterpretation, CourtDecision, DecisionSection, Statute

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "raw"


def _load(rel):
    return json.loads((FIXTURES / rel).read_text(encoding="utf-8"))


def test_judgment_normalizes_sections_and_refs():
    doc = normalize("aihub_judgment_v1", _load("aihub_judgment/judgment_001.json"), dataset="aihub_judgment", file="aihub_judgment/judgment_001.json")
    assert isinstance(doc, CourtDecision)
    assert doc.case_number == "2019가단12345" and doc.court == "서울중앙지방법원"
    assert doc.decision_date == date(2020, 3, 15)
    assert set(doc.sections) >= {DecisionSection.facts, DecisionSection.reasoning, DecisionSection.holding, DecisionSection.issues}
    assert "주택임대차보호법 제3조의3" in doc.referenced_statutes
    assert doc.legal_domain == "민사"
    assert doc.text.startswith("서울중앙지방법원 2019가단12345")


def test_statute_normalizes_hierarchy():
    doc = normalize("statute_json_v1", _load("statutes/housing_lease_act.json"), dataset="statutes", file="statutes/housing_lease_act.json")
    assert isinstance(doc, Statute)
    art = {a.number: a for a in doc.articles}["제3조의3"]
    assert art.paragraphs[1].items[0].number == "제1호"
    assert doc.effective_date == date(2020, 12, 10)
    assert doc.source.record_key == "주택임대차보호법@법률 제17363호"


def test_admin_interpretation_normalizes():
    doc = normalize("admin_interpretation_v1", _load("admin_interpretations/interp_001.json"), dataset="admin_interpretations", file="x.json")
    assert isinstance(doc, AdminInterpretation) and doc.reference_number == "19-0123"


def test_missing_case_number_raises():
    payload = _load("aihub_judgment/judgment_001.json")
    payload["info"].pop("caseNo")
    with pytest.raises(NormalizationError):
        normalize("aihub_judgment_v1", payload, dataset="d", file="f")


def test_unknown_schema_raises():
    with pytest.raises(NormalizationError):
        normalize("nope", {}, dataset="d", file="f")
```

**Step 2: Run, expect ImportError.**

**Step 3: Implement**

`base.py`:

```python
class NormalizationError(ValueError):
    """Raised when a raw record cannot be turned into a canonical document."""


def as_text(value) -> str:
    """AI Hub fields are sometimes strings and sometimes lists of strings."""
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(as_text(v) for v in value if v is not None).strip()
    return str(value).strip()
```

`aihub_judgment.py` — `normalize_aihub_judgment(payload, dataset, file) -> CourtDecision`:
- `info = payload.get("info", {})`; require `caseNo` and `courtNm`, else `NormalizationError`.
- Sections: `issues ← mentionedItems.rqestObjet`, `facts ← facts.bsisFacts`, `reasoning ← dcss.courtDcss`, `holding ← close.cnclsns`. Include only non-empty.
- `legal_domain ← info.detailField` when present, else map `caseField` `"1"→"민사", "2"→"형사", "3"→"행정"` else `None`.
- `referenced_statutes` = `[r.render() for r in extract_statute_refs(all section text)]`, `referenced_cases` = `extract_case_refs(all section text)` minus the document's own case number.
- `title = f"{court} {case_number} {caseNm}".strip()`.
- `text` = title line, then for each present section in the order issues, facts, reasoning, holding, summary: `"[{korean label}]\n{text}"` joined by blank lines. Labels: 판시사항/사실관계/판단/주문/요지.
- `id = make_document_id(DocumentType.court_decision, dataset, case_number)`; `source = SourceRef(dataset, file, record_key=case_number)`.

`statute_json.py` — `normalize_statute_json(payload, dataset, file) -> Statute`:
- require `law_name` and non-empty `articles`.
- normalize article numbers with `normalize_article`; paragraph numbers `제N항`, item numbers `제N호` (accept bare digits and prefix them).
- `record_key = f"{law_name}@{version or 'unversioned'}"`.
- `text` = `law_name` line, then for each article: `"{number}({title}) {text}"`, then paragraphs `"  {para.number} {para.text}"`, then items `"    {item.number} {item.text}"`.
- `title = law_name`.

`admin_interpretation.py` — straightforward; `record_key = reference_number`; `text` = subject line, `[질의]` question, `[회답]` interpretation, `[관련 법령]` joined basis.

`normalize/__init__.py`:

```python
_NORMALIZERS = {
    "aihub_judgment_v1": normalize_aihub_judgment,
    "statute_json_v1": normalize_statute_json,
    "admin_interpretation_v1": normalize_admin_interpretation,
}

def normalize(schema: str, payload: dict, *, dataset: str, file: str) -> LegalDocument:
    try:
        fn = _NORMALIZERS[schema]
    except KeyError:
        raise NormalizationError(f"unknown schema {schema!r}") from None
    return fn(payload, dataset=dataset, file=file)
```

**Step 4: Run, expect 5 passed. Commit**

```bash
git add src/legal_rag/ingestion/normalize tests/legal_rag/unit/test_ingestion_normalize.py
git commit -m "feat: normalize AI Hub judgments, statutes, and interpretations to canonical documents"
```

---

### Task 9: Validation, deduplication, manifest, and the ingestion pipeline

**Files:**
- Create: `src/legal_rag/ingestion/validate.py`, `dedupe.py`, `manifest.py`, `pipeline.py`, `corpus.py`
- Modify: `src/legal_rag/ingestion/__init__.py` (add `PIPELINE_VERSION = "2026.09.07"`)
- Test: `tests/legal_rag/unit/test_ingestion_pipeline.py`

**Step 1: Failing tests**

```python
import json
from pathlib import Path

from legal_rag.config import load_settings
from legal_rag.ingestion.corpus import Corpus
from legal_rag.ingestion.pipeline import run_ingestion

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "raw"


def _settings(tmp_path: Path):
    return load_settings(profile="default", overrides={
        "paths": {"root": str(tmp_path), "raw_dir": str(FIXTURES)},
        "corpus": {"version": "test"},
    })


def test_pipeline_writes_documents_chunks_and_manifest(tmp_path: Path):
    report = run_ingestion(_settings(tmp_path))
    out = tmp_path / "data" / "processed" / "test"
    assert (out / "documents.jsonl").exists() and (out / "chunks.jsonl").exists()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["corpus_version"] == "test"
    assert manifest["counts"]["normalized"] == 5
    assert manifest["counts"]["unrecognized"] == 1
    assert any(f["file"].endswith("judgment_001.json") for f in manifest["raw_files"])
    assert report.documents == 5 and report.chunks > 5


def test_pipeline_is_byte_reproducible(tmp_path: Path):
    run_ingestion(_settings(tmp_path / "a"))
    run_ingestion(_settings(tmp_path / "b"))
    for name in ("documents.jsonl", "chunks.jsonl"):
        assert (tmp_path / "a/data/processed/test" / name).read_bytes() == (tmp_path / "b/data/processed/test" / name).read_bytes()


def test_duplicates_are_dropped(tmp_path: Path):
    raw = tmp_path / "raw" / "statutes"
    raw.mkdir(parents=True)
    src = (FIXTURES / "statutes" / "civil_act_excerpt.json").read_text(encoding="utf-8")
    (raw / "a.json").write_text(src, encoding="utf-8")
    (raw / "b.json").write_text(src, encoding="utf-8")
    settings = load_settings(profile="default", overrides={"paths": {"root": str(tmp_path), "raw_dir": str(tmp_path / "raw")}, "corpus": {"version": "t"}})
    report = run_ingestion(settings)
    assert report.documents == 1 and report.deduplicated == 1


def test_corpus_reader_round_trips(tmp_path: Path):
    run_ingestion(_settings(tmp_path))
    corpus = Corpus.load(tmp_path / "data" / "processed" / "test")
    docs = list(corpus.documents())
    chunks = list(corpus.chunks())
    assert len(docs) == 5 and all(c.document_id in {d.id for d in docs} for c in chunks)
    assert corpus.document(chunks[0].document_id).id == chunks[0].document_id
```

Note: `run_ingestion` also chunks (Task 10 provides the chunkers). Write Task 10 before running these tests to green; commit the pipeline together with the chunkers if needed, or stub `chunk_document` to return one fallback chunk per document and replace it in Task 10.

**Step 2: Implement**

`validate.py`:

```python
@dataclass(frozen=True)
class ValidationFailure:
    document_id: str
    reason: str

def validate(doc: LegalDocument) -> ValidationFailure | None:
    if not doc.title.strip() or not doc.text.strip():
        return ValidationFailure(doc.id, "empty title or text")
    if isinstance(doc, Statute) and not doc.articles:
        return ValidationFailure(doc.id, "statute without articles")
    if isinstance(doc, CourtDecision) and (not doc.case_number or not doc.sections):
        return ValidationFailure(doc.id, "decision without case number or sections")
    if isinstance(doc, AdminInterpretation) and (not doc.question or not doc.interpretation):
        return ValidationFailure(doc.id, "interpretation without question or answer")
    return None
```

`dedupe.py`:

```python
class Deduplicator:
    def __init__(self):
        self._keys: set[tuple[str, str]] = set()
        self._hashes: set[str] = set()

    def is_duplicate(self, doc: LegalDocument) -> str | None:
        key = (doc.source.dataset, doc.source.record_key)
        if key in self._keys:
            return "duplicate record key"
        digest = hashlib.sha256(re.sub(r"\s+", " ", doc.text).strip().encode("utf-8")).hexdigest()
        if digest in self._hashes:
            return "duplicate text"
        self._keys.add(key); self._hashes.add(digest)
        return None
```

`manifest.py` — `Manifest` Pydantic model (not frozen): `corpus_version`, `pipeline_version`, `config_hash`, `created_at` (ISO, UTC), `raw_files: list[{file, size, sha256}]`, `counts: {records, recognized, unrecognized, normalized, normalization_errors, validation_failures, deduplicated, documents, chunks}`, `dropped: list[{file, index, stage, reason}]`. `write(path)` / `read(path)`.

`pipeline.py`:

```python
@dataclass(frozen=True)
class IngestionReport:
    documents: int
    chunks: int
    deduplicated: int
    unrecognized: int
    output_dir: Path

def run_ingestion(settings: Settings) -> IngestionReport:
    raw_dir = settings.paths.root / settings.paths.raw_dir if not settings.paths.raw_dir.is_absolute() else settings.paths.raw_dir
    out_dir = settings.processed_corpus_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest.new(settings)
    dedupe = Deduplicator()
    documents: list[LegalDocument] = []
    seen_files: dict[str, tuple[int, str]] = {}
    for record in discover_raw_records(raw_dir):
        seen_files.setdefault(record.file, ((raw_dir / record.file).stat().st_size, record.file_sha256))
        manifest.counts.records += 1
        schema = detect_schema(record.payload)
        if schema is None:
            manifest.counts.unrecognized += 1; manifest.drop(record, "detect", "unrecognized layout"); continue
        manifest.counts.recognized += 1
        try:
            doc = normalize(schema, record.payload, dataset=record.dataset, file=record.file)
        except NormalizationError as exc:
            manifest.counts.normalization_errors += 1; manifest.drop(record, "normalize", str(exc)); continue
        manifest.counts.normalized += 1
        if failure := validate(doc):
            manifest.counts.validation_failures += 1; manifest.drop(record, "validate", failure.reason); continue
        if reason := dedupe.is_duplicate(doc):
            manifest.counts.deduplicated += 1; manifest.drop(record, "dedupe", reason); continue
        documents.append(doc)
    documents.sort(key=lambda d: d.id)
    chunks = [c for doc in documents for c in chunk_document(doc, corpus_version=settings.corpus.version)]
    _write_jsonl(out_dir / "documents.jsonl", documents)
    _write_jsonl(out_dir / "chunks.jsonl", chunks)
    manifest.raw_files = [{"file": f, "size": s, "sha256": h} for f, (s, h) in sorted(seen_files.items())]
    manifest.counts.documents = len(documents); manifest.counts.chunks = len(chunks)
    manifest.write(out_dir / "manifest.json")
    return IngestionReport(len(documents), len(chunks), manifest.counts.deduplicated, manifest.counts.unrecognized, out_dir)
```

`_write_jsonl` writes `model_dump_json()` per line with `\n`, UTF-8, no trailing spaces. Sorting documents by id and chunks by document order makes the output reproducible. `created_at` lives only in the manifest, which the reproducibility test does not compare.

`corpus.py` — `Corpus(dir)` with `documents()` (TypeAdapter over the union, streamed), `chunks()`, `document(id)` (lazy index built on first call), `manifest()`.

**Step 3: Run tests after Task 10; expect 4 passed. Commit**

```bash
git add src/legal_rag/ingestion tests/legal_rag/unit/test_ingestion_pipeline.py
git commit -m "feat: add validation, deduplication, manifest, and the reproducible ingestion pipeline"
```

---

### Task 10: Legal-aware chunkers

**Files:**
- Create: `src/legal_rag/chunking/base.py`, `statute.py`, `decision.py`, `fallback.py`, `router.py`
- Modify: `src/legal_rag/chunking/__init__.py` (export `chunk_document`)
- Test: `tests/legal_rag/unit/test_chunking.py`

**Step 1: Failing tests**

```python
import json
from pathlib import Path

from legal_rag.chunking import chunk_document
from legal_rag.chunking.fallback import window_sentences
from legal_rag.ingestion.normalize import normalize

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "raw"


def _doc(schema, rel, dataset):
    return normalize(schema, json.loads((FIXTURES / rel).read_text(encoding="utf-8")), dataset=dataset, file=rel)


def test_short_article_is_one_chunk_and_long_article_splits_by_paragraph():
    statute = _doc("statute_json_v1", "statutes/housing_lease_act.json", "statutes")
    chunks = chunk_document(statute, corpus_version="v1")
    paths = [c.path for c in chunks]
    assert "art:제3조" in paths
    assert "art:제3조의3/para:제1항" in paths and "art:제3조의3/para:제2항" in paths
    assert all(c.metadata.law_name == "주택임대차보호법" for c in chunks)
    para2 = next(c for c in chunks if c.path == "art:제3조의3/para:제2항")
    assert "제1호 신청의 취지 및 이유" in para2.text
    assert para2.text.startswith("주택임대차보호법 제3조의3(임차권등기명령) 제2항")
    assert [c.order for c in chunks] == list(range(len(chunks)))


def test_decision_chunks_by_section_and_windows_long_sections():
    decision = _doc("aihub_judgment_v1", "aihub_judgment/judgment_002.json", "aihub_judgment")
    chunks = chunk_document(decision, corpus_version="v1")
    paths = [c.path for c in chunks]
    assert "sec:facts" in paths and "sec:holding" in paths
    assert "sec:reasoning/1" in paths and "sec:reasoning/2" in paths
    reasoning = [c for c in chunks if c.path.startswith("sec:reasoning")]
    assert all(c.metadata.case_number == "2018다12345" and c.metadata.section == "reasoning" for c in reasoning)
    assert reasoning[0].text.startswith("대법원 2018다12345")


def test_admin_interpretation_chunks_question_and_answer_together_when_short():
    doc = _doc("admin_interpretation_v1", "admin_interpretations/interp_001.json", "admin_interpretations")
    chunks = chunk_document(doc, corpus_version="v1")
    assert [c.path for c in chunks] == ["sec:qa"]
    assert chunks[0].metadata.authority == "법제처"


def test_window_sentences_overlaps_by_one_sentence():
    sentences = [f"문장{i}입니다." for i in range(6)]
    windows = window_sentences(sentences, max_chars=20)
    assert windows[0] == "문장0입니다. 문장1입니다."
    assert windows[1].startswith("문장1입니다.")
```

**Step 2: Run, expect ImportError.**

**Step 3: Implement**

`base.py`:

```python
STATUTE_MAX_CHARS = 800
DECISION_MAX_CHARS = 1200
INTERPRETATION_MAX_CHARS = 1200

def make_chunk(doc, *, path, order, text, corpus_version, **meta) -> Chunk:
    return Chunk(
        id=f"{doc.id}#{path}", document_id=doc.id, document_type=doc.document_type, path=path, order=order,
        text=text, title=doc.title,
        metadata=ChunkMetadata(dataset=doc.source.dataset, corpus_version=corpus_version,
                               legal_domain=doc.legal_domain, effective_date=doc.effective_date, **meta),
    )
```

`fallback.py`:

```python
def window_sentences(sentences: list[str], max_chars: int) -> list[str]:
    windows, current = [], []
    for s in sentences:
        if current and len(" ".join(current + [s])) > max_chars:
            windows.append(" ".join(current))
            current = [current[-1], s]          # one-sentence overlap
        else:
            current.append(s)
    if current:
        windows.append(" ".join(current))
    return windows

def chunk_text_windows(doc, text, *, path_prefix, start_order, corpus_version, max_chars, **meta) -> list[Chunk]
```
`chunk_text_windows` splits with `split_sentences`, windows, and emits `f"{path_prefix}/{n}"` (1-based) when more than one window, else `path_prefix` alone. Every chunk text starts with the header line passed in `meta["_header"]` (pop it before building metadata).

`statute.py` — `chunk_statute(doc, corpus_version)`:
- For each article: `header = f"{law_name} {number}({title})"` (omit parens when no title). `article_body = article.text` plus rendered paragraphs (`"{para.number} {para.text}"` + items `"{item.number} {item.text}"`).
- If `len(header + body) <= STATUTE_MAX_CHARS` → one chunk `art:{number}`.
- Else one chunk per paragraph `art:{number}/para:{para.number}` with text `f"{header} {para.number}\n{para.text}\n{items}"`; if a paragraph exceeds the limit, window it under `art:.../para:.../{n}`. If the article has body text but no paragraphs and exceeds the limit, window under `art:{number}/{n}`.
- Metadata: `law_name`, `article`, `paragraph` (None for whole-article chunks).

`decision.py` — `chunk_decision(doc, corpus_version)`:
- Section order: issues, facts, reasoning, holding, summary. Korean labels as in Task 8.
- `header = f"{court} {case_number} ({decision_date or '날짜 미상'}) — {label}"`.
- One chunk `sec:{name}` when within `DECISION_MAX_CHARS`, else windows `sec:{name}/{n}`.
- Metadata: `court`, `case_number`, `decision_date`, `section`.

`router.py`:

```python
def chunk_document(doc: LegalDocument, *, corpus_version: str) -> list[Chunk]:
    if isinstance(doc, Statute): return chunk_statute(doc, corpus_version)
    if isinstance(doc, CourtDecision): return chunk_decision(doc, corpus_version)
    if isinstance(doc, AdminInterpretation): return chunk_interpretation(doc, corpus_version)
    return chunk_fallback(doc, corpus_version)
```

`chunk_interpretation` lives in `decision.py`'s sibling `interpretation.py`: one chunk `sec:qa` with `header = f"{authority} {reference_number} ({date}) — {subject}"` and text `[질의] ... [회답] ...` when within the limit, else two chunks `sec:question` and `sec:interpretation` (windowed if needed). Metadata: `authority`, `reference_number`.

**Step 4: Run all tests, expect Task 9 and Task 10 tests green. Commit**

```bash
git add src/legal_rag/chunking tests/legal_rag/unit/test_chunking.py
git commit -m "feat: add legal-aware chunkers for statutes, decisions, and interpretations"
```

---

### Task 11: `scripts/ingest.py`

**Files:**
- Create: `scripts/ingest.py`
- Test: `tests/legal_rag/integration/test_scripts_ingest.py`

**Step 1: Failing test**

```python
import subprocess, sys
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "raw"
ROOT = Path(__file__).resolve().parents[3]


def test_ingest_cli_runs(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "ingest.py"), "--profile", "default", "--root", str(tmp_path),
         "--raw-dir", str(FIXTURES), "--corpus-version", "cli"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "documents=5" in result.stdout
    assert (tmp_path / "data/processed/cli/manifest.json").exists()
```

**Step 2: Implement with Typer**

```python
#!/usr/bin/env python
"""Normalize raw legal data into a reproducible processed corpus."""
from pathlib import Path
import typer
from legal_rag.config import load_settings
from legal_rag.ingestion.pipeline import run_ingestion

app = typer.Typer(add_completion=False)

@app.command()
def main(profile: str = typer.Option(None), root: Path = typer.Option(None), raw_dir: Path = typer.Option(None),
         corpus_version: str = typer.Option(None)):
    overrides = {}
    if root: overrides.setdefault("paths", {})["root"] = str(root)
    if raw_dir: overrides.setdefault("paths", {})["raw_dir"] = str(raw_dir)
    if corpus_version: overrides["corpus"] = {"version": corpus_version}
    report = run_ingestion(load_settings(profile=profile, overrides=overrides))
    typer.echo(f"documents={report.documents} chunks={report.chunks} deduplicated={report.deduplicated} "
               f"unrecognized={report.unrecognized} output={report.output_dir}")

if __name__ == "__main__":
    app()
```

Because scripts run outside the package, add `sys.path` handling? No: `uv sync` installs `legal_rag` in editable mode, so `uv run python scripts/ingest.py` works. The test uses `sys.executable` which is the `.venv` interpreter under `uv run pytest`.

**Step 3: Run, expect 1 passed. Commit**

```bash
git add scripts/ingest.py tests/legal_rag/integration/test_scripts_ingest.py
git commit -m "feat: add the ingest CLI"
```

V0.1 exit criterion met: fixtures ingest reproducibly.

---

## V0.2 — Retrieval baseline

### Task 12: Embedder Protocol with fake and Ollama backends

**Files:**
- Create: `src/legal_rag/embeddings/base.py`, `fake.py`, `ollama.py`; export from `__init__.py`
- Test: `tests/legal_rag/unit/test_embeddings.py`

**Step 1: Failing tests**

```python
import httpx
import pytest

from legal_rag.embeddings import FakeEmbedder, OllamaEmbedder


def test_fake_embedder_is_deterministic_and_unit_norm():
    e = FakeEmbedder(dimension=32)
    a, b = e.embed_query("보증금 반환"), e.embed_query("보증금 반환")
    assert a == b and len(a) == 32
    assert abs(sum(x * x for x in a) - 1.0) < 1e-6
    assert e.embed_query("불법행위 손해배상") != a


def test_fake_embedder_shares_tokens_between_similar_texts():
    e = FakeEmbedder(dimension=32)
    dot = lambda x, y: sum(p * q for p, q in zip(x, y))
    q = e.embed_query("보증금 반환")
    assert dot(q, e.embed_query("임대차 보증금 반환 청구")) > dot(q, e.embed_query("교통사고 과실 비율"))


def test_ollama_embedder_batches_and_parses(monkeypatch):
    calls = []
    def handler(request: httpx.Request):
        body = request.read().decode()
        calls.append(body)
        n = body.count('"') and len(httpx.Response(200).json if False else []) or 0
        import json
        payload = json.loads(body)
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2] for _ in payload["input"]]})
    transport = httpx.MockTransport(handler)
    e = OllamaEmbedder(model="bge-m3", base_url="http://test", batch_size=2, transport=transport)
    vectors = e.embed_documents(["a", "b", "c"])
    assert len(vectors) == 3 and len(calls) == 2
    assert e.name == "ollama:bge-m3"
```

(Remove the stray `n = ...` line when writing the test; it is noise.)

**Step 2: Implement**

`base.py`: the Protocol from design section 4 and `def slug(name: str) -> str` returning `re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")`.

`fake.py`:

```python
class FakeEmbedder:
    def __init__(self, dimension: int = 64):
        self.dimension = dimension
        self.name = f"fake:{dimension}"

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for token in RegexTokenizer().tokenize(text):
            h = int.from_bytes(hashlib.sha1(token.encode("utf-8")).digest()[:8], "big")
            vec[h % self.dimension] += 1.0 if (h >> 63) == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts): return [self._embed(t) for t in texts]
    def embed_query(self, text): return self._embed(text)
```

This depends on `RegexTokenizer` from Task 13; implement Task 13's tokenizer first or in the same commit. The sign flip must not depend on `h % dimension` so identical tokens always add in the same direction, which the test relies on.

`ollama.py`:

```python
class OllamaEmbedder:
    def __init__(self, model: str, base_url: str, batch_size: int = 16, timeout: float = 120.0, transport=None):
        self.model, self.batch_size = model, batch_size
        self.name = f"ollama:{model}"
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.embed_query("차원 확인"))
        return self._dimension

    def embed_documents(self, texts):
        out = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._call(texts[i:i + self.batch_size]))
        return out

    def embed_query(self, text): return self._call([text])[0]

    def _call(self, inputs):
        r = self._client.post("/api/embed", json={"model": self.model, "input": inputs})
        r.raise_for_status()
        return r.json()["embeddings"]
```

**Step 3: Run, expect 3 passed. Commit** (`feat: add embedder protocol with fake and Ollama backends`).

---

### Task 13: Korean tokenizer and BM25 statistics

**Files:**
- Create: `src/legal_rag/retrieval/tokenizer.py`, `src/legal_rag/retrieval/sparse.py`
- Test: `tests/legal_rag/unit/test_tokenizer.py`, `tests/legal_rag/unit/test_bm25.py`

**Step 1: Failing tokenizer tests**

```python
from legal_rag.retrieval.tokenizer import RegexTokenizer


def test_preserves_legal_identifiers():
    toks = RegexTokenizer().tokenize("민법 제750조와 주택임대차보호법 제3조의2, 2018다12345 판결")
    assert "제750조" in toks and "제3조의2" in toks and "2018다12345" in toks


def test_emits_hangul_bigrams_and_whole_words():
    toks = RegexTokenizer().tokenize("보증금반환")
    assert "보증금반환" in toks and "보증" in toks and "증금" in toks and "금반" in toks and "반환" in toks


def test_lowercases_ascii_and_drops_punctuation():
    assert RegexTokenizer().tokenize("RAG, Qdrant!") == ["rag", "qdrant"]
```

**Step 2: Implement `tokenizer.py`**

```python
_SPECIAL = re.compile(rf"({CASE_RE.pattern}|제\s*\d+\s*조(?:\s*의\s*\d+)?|제\d+항|제\d+호)")
_WORD = re.compile(r"[가-힣]+|[a-z0-9]+")

class RegexTokenizer:
    name = "regex"

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        pos = 0
        for m in _SPECIAL.finditer(text):
            tokens += self._words(text[pos:m.start()])
            tokens.append(re.sub(r"\s+", "", m.group(0)))
            pos = m.end()
        tokens += self._words(text[pos:])
        return tokens

    def _words(self, segment: str) -> list[str]:
        out = []
        for w in _WORD.findall(segment.lower()):
            out.append(w)
            if re.fullmatch(r"[가-힣]{3,}", w):
                out += [w[i:i + 2] for i in range(len(w) - 1)]
        return out
```

`CASE_RE.pattern` contains a capture group; use `CASE_RE.pattern.replace("(", "(?:", 1)` or rebuild without the group. Also add `KiwiTokenizer` guarded by `try: from kiwipiepy import Kiwi` that yields nouns/verbs plus the same special tokens; if the import fails, `build_tokenizer("kiwi")` raises a clear `RuntimeError`.

**Step 3: Failing BM25 tests**

```python
from legal_rag.retrieval.sparse import BM25Stats, term_id


def test_fit_and_weights():
    docs = {"a": ["보증금", "반환", "보증금"], "b": ["불법행위", "손해"], "c": ["보증금", "손해"]}
    stats = BM25Stats.fit(docs, k1=1.2, b=0.75)
    assert stats.n_docs == 3 and stats.avg_len == (3 + 2 + 2) / 3
    assert stats.idf("보증금") < stats.idf("불법행위")          # rarer term, higher idf
    doc_vec = stats.document_vector(["보증금", "반환", "보증금"])
    assert term_id("보증금") in dict(zip(doc_vec.indices, doc_vec.values))
    query_vec = stats.query_vector(["보증금", "없는단어"])
    assert query_vec.indices == [term_id("보증금")]              # unknown terms dropped


def test_round_trip_json(tmp_path):
    stats = BM25Stats.fit({"a": ["x", "y"], "b": ["y"]}, k1=1.2, b=0.75)
    stats.save(tmp_path / "idf.json")
    loaded = BM25Stats.load(tmp_path / "idf.json")
    assert loaded.idf("y") == stats.idf("y") and loaded.avg_len == stats.avg_len
```

**Step 4: Implement `sparse.py`**

```python
def term_id(term: str) -> int:
    return int.from_bytes(hashlib.sha1(term.encode("utf-8")).digest()[:4], "big")

@dataclass
class SparseVec:
    indices: list[int]
    values: list[float]

class BM25Stats:
    def __init__(self, df: dict[str, int], n_docs: int, avg_len: float, k1: float, b: float): ...

    @classmethod
    def fit(cls, tokenized: dict[str, list[str]], *, k1: float, b: float) -> "BM25Stats":
        df = Counter()
        total = 0
        for tokens in tokenized.values():
            total += len(tokens)
            df.update(set(tokens))
        return cls(dict(df), len(tokenized), total / max(len(tokenized), 1), k1, b)

    def idf(self, term: str) -> float:
        d = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - d + 0.5) / (d + 0.5))

    def document_vector(self, tokens: list[str]) -> SparseVec:
        tf = Counter(tokens); dl = len(tokens)
        pairs = {}
        for term, f in tf.items():
            w = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg_len))
            pairs[term_id(term)] = pairs.get(term_id(term), 0.0) + w
        idx = sorted(pairs); return SparseVec(idx, [pairs[i] for i in idx])

    def query_vector(self, tokens: list[str]) -> SparseVec:
        pairs = {}
        for term in set(tokens):
            if term in self.df:
                pairs[term_id(term)] = self.idf(term)
        idx = sorted(pairs); return SparseVec(idx, [pairs[i] for i in idx])

    def save(self, path): path.write_text(json.dumps({"df": self.df, "n_docs": ..., "avg_len": ..., "k1": ..., "b": ...}, ensure_ascii=False), encoding="utf-8")
    @classmethod
    def load(cls, path): ...
```

**Step 5: Run, expect 5 passed. Commit** (`feat: add Korean regex tokenizer and BM25 sparse statistics`).

---

### Task 14: Qdrant store, filters, dense and sparse retrievers

**Files:**
- Create: `src/legal_rag/retrieval/filters.py`, `store.py`, `dense.py`, `sparse_retriever.py`
- Test: `tests/legal_rag/unit/test_store.py`

**Step 1: Failing tests** (embedded Qdrant in `tmp_path`)

```python
from datetime import date
from pathlib import Path

from legal_rag.embeddings import FakeEmbedder
from legal_rag.retrieval.dense import DenseRetriever
from legal_rag.retrieval.sparse import BM25Stats
from legal_rag.retrieval.sparse_retriever import SparseRetriever
from legal_rag.retrieval.store import VectorStore
from legal_rag.retrieval.tokenizer import RegexTokenizer
from legal_rag.schemas import Chunk, ChunkMetadata, DocumentType, RetrievalFilters


def _chunks():
    mk = lambda i, t, dt, **m: Chunk(id=f"d{i}#p", document_id=f"d{i}", document_type=dt, path="p", order=0, text=t, title="t",
                                    metadata=ChunkMetadata(dataset="f", corpus_version="v", **m))
    return [
        mk(1, "임대차 보증금 반환 청구", DocumentType.statute, law_name="주택임대차보호법", article="제3조의2"),
        mk(2, "불법행위 손해배상 책임", DocumentType.statute, law_name="민법", article="제750조"),
        mk(3, "보증금 반환 판결 이유", DocumentType.court_decision, court="대법원", case_number="2018다12345", decision_date=date(2019, 5, 30)),
    ]


def _build(tmp_path: Path):
    chunks = _chunks()
    embedder, tok = FakeEmbedder(32), RegexTokenizer()
    stats = BM25Stats.fit({c.id: tok.tokenize(c.text) for c in chunks}, k1=1.2, b=0.75)
    store = VectorStore.embedded(tmp_path / "q", collection="test", dimension=32)
    store.recreate()
    store.upsert(chunks, dense=embedder.embed_documents([c.text for c in chunks]),
                 sparse=[stats.document_vector(tok.tokenize(c.text)) for c in chunks])
    return store, embedder, tok, stats


def test_dense_retrieval_ranks_similar_text_first(tmp_path):
    store, embedder, *_ = _build(tmp_path)
    hits = DenseRetriever(store, embedder).retrieve("보증금 반환", k=3)
    assert hits[0].chunk.document_id in {"d1", "d3"} and hits[0].retriever == "dense"


def test_sparse_retrieval_matches_exact_article(tmp_path):
    store, _, tok, stats = _build(tmp_path)
    hits = SparseRetriever(store, tok, stats).retrieve("민법 제750조", k=3)
    assert hits[0].chunk.metadata.article == "제750조" and hits[0].retriever == "sparse"


def test_filters_restrict_document_type_and_date(tmp_path):
    store, embedder, *_ = _build(tmp_path)
    r = DenseRetriever(store, embedder)
    only_cases = r.retrieve("보증금", k=3, filters=RetrievalFilters(document_types=[DocumentType.court_decision]))
    assert {h.chunk.document_type for h in only_cases} == {DocumentType.court_decision}
    none = r.retrieve("보증금", k=3, filters=RetrievalFilters(decided_after=date(2020, 1, 1)))
    assert none == []


def test_store_round_trips_chunk_payload(tmp_path):
    store, embedder, *_ = _build(tmp_path)
    hit = DenseRetriever(store, embedder).retrieve("손해배상", k=1)[0]
    assert hit.chunk.metadata.law_name == "민법" and hit.chunk.id == "d2#p"
```

**Step 2: Implement**

`filters.py` — `to_qdrant_filter(filters: RetrievalFilters | None)`; import Qdrant models here only via `store.py` re-export? Rule says only `store.py` imports `qdrant_client`. So `filters.py` produces a neutral structure `list[Condition]` where `Condition = ("match", key, value) | ("range", key, gte, lte)`, and `store.py` converts it. Keys: `document_type`, `legal_domain`, `court`, `law_name`, `decision_date_int` (range from `decided_after`/`decided_before` as `YYYYMMDD` ints), `effective_date_int` (`effective_on` → `lte`).

`store.py`:

```python
class VectorStore:
    DENSE = "dense"; SPARSE = "bm25"

    def __init__(self, client, collection: str, dimension: int): ...
    @classmethod
    def embedded(cls, path: Path, collection: str, dimension: int): return cls(QdrantClient(path=str(path)), collection, dimension)
    @classmethod
    def server(cls, url: str, api_key: str | None, collection: str, dimension: int): ...

    def recreate(self):
        self.client.recreate_collection(...)  # or delete+create; named vectors: {"dense": VectorParams(size, Distance.COSINE)}, sparse_vectors_config={"bm25": SparseVectorParams()}
        self.client.create_payload_index(self.collection, "document_type", "keyword")   # plus law_name, court, legal_domain keyword; decision_date_int, effective_date_int integer

    def upsert(self, chunks, dense, sparse): points in batches of 128; id = uuid5(NAMESPACE_URL, chunk.id); payload = chunk.model_dump(mode="json") | {"decision_date_int": ..., "effective_date_int": ...}
    def search_dense(self, vector, k, filters) -> list[tuple[Chunk, float]]
    def search_sparse(self, sparse: SparseVec, k, filters) -> list[tuple[Chunk, float]]
    def count(self) -> int
```

`search_*` use `client.query_points(collection, query=..., using=..., limit=k, query_filter=..., with_payload=True)`, rebuild `Chunk.model_validate({k: v for k, v in payload.items() if k not in ("decision_date_int", "effective_date_int")})`.

`dense.py`:

```python
class DenseRetriever:
    name = "dense"
    def __init__(self, store, embedder): ...
    def retrieve(self, query, k, filters=None):
        return [ScoredChunk(chunk=c, score=s, retriever="dense") for c, s in self.store.search_dense(self.embedder.embed_query(query), k, filters)]
```

`sparse_retriever.py` mirrors it with tokenizer + stats. Note the earlier design named this file `sparse.py`; `sparse.py` keeps BM25 statistics and `sparse_retriever.py` holds the retriever.

**Step 3: Run, expect 4 passed. Commit** (`feat: add embedded Qdrant store with dense and sparse retrievers and metadata filters`).

---

### Task 15: RRF fusion, hybrid retriever, noop reranker

**Files:**
- Create: `src/legal_rag/retrieval/fusion.py`, `hybrid.py`; `src/legal_rag/reranking/base.py`, `noop.py`
- Test: `tests/legal_rag/unit/test_fusion.py`

**Step 1: Failing tests**

```python
from legal_rag.reranking import NoopReranker
from legal_rag.retrieval.fusion import RRFFuser
from legal_rag.retrieval.hybrid import HybridRetriever
from legal_rag.schemas import ScoredChunk
from tests.legal_rag.unit.test_store import _chunks


def _sc(chunk, score, name): return ScoredChunk(chunk=chunk, score=score, retriever=name)


def test_rrf_rewards_chunks_present_in_both_lists():
    c1, c2, c3 = _chunks()
    dense = [_sc(c1, 0.9, "dense"), _sc(c2, 0.8, "dense")]
    sparse = [_sc(c3, 5.0, "sparse"), _sc(c1, 4.0, "sparse")]
    fused = RRFFuser(rrf_k=60).fuse([dense, sparse], k=3)
    assert fused[0].chunk.id == c1.id and fused[0].retriever == "fused"
    assert abs(fused[0].score - (1 / 61 + 1 / 62)) < 1e-9
    assert fused[0].chunk.metadata is c1.metadata or fused[0].chunk == c1


def test_rrf_applies_leg_weights():
    c1, c2, _ = _chunks()
    fused = RRFFuser(rrf_k=60, weights={"dense": 0.0, "sparse": 1.0}).fuse([[_sc(c1, 1, "dense")], [_sc(c2, 1, "sparse")]], k=2)
    assert fused[0].chunk.id == c2.id


class _Stub:
    def __init__(self, name, hits): self.name, self.hits = name, hits
    def retrieve(self, query, k, filters=None): return self.hits[:k]


def test_hybrid_runs_every_query_through_every_leg_and_reranks():
    c1, c2, c3 = _chunks()
    dense = _Stub("dense", [_sc(c1, 0.9, "dense")])
    sparse = _Stub("sparse", [_sc(c2, 1.0, "sparse")])
    hybrid = HybridRetriever([dense, sparse], RRFFuser(60), NoopReranker(), k_leg=5, k_fused=5)
    hits = hybrid.retrieve_many(["q1", "q2"], k=2)
    assert {h.chunk.id for h in hits} == {c1.id, c2.id} and all(h.retriever == "reranked" for h in hits)
    assert hits[0].chunk.metadata.model_dump() and "fused_score" in hybrid.last_trace["fused"][0]
```

Adjust the final assertion to what the implementation exposes: `HybridRetriever.last_trace: dict[str, list[tuple[str, float]]]` keyed `"dense:q1"`, `"sparse:q1"`, ..., `"fused"`, `"reranked"`.

**Step 2: Implement**

`fusion.py`:

```python
class RRFFuser:
    def __init__(self, rrf_k: int = 60, weights: dict[str, float] | None = None): ...
    def fuse(self, ranked_lists, k):
        scores: dict[str, float] = {}; chunks: dict[str, Chunk] = {}
        for ranked in ranked_lists:
            for rank, hit in enumerate(ranked, start=1):
                w = self.weights.get(hit.retriever, 1.0)
                scores[hit.chunk.id] = scores.get(hit.chunk.id, 0.0) + w / (self.rrf_k + rank)
                chunks.setdefault(hit.chunk.id, hit.chunk)
        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [ScoredChunk(chunk=chunks[cid], score=s, retriever="fused") for cid, s in ordered]
```

`reranking/base.py` Protocol; `noop.py`:

```python
class NoopReranker:
    name = "noop"
    def rerank(self, query, candidates, k):
        return [c.model_copy(update={"retriever": "reranked"}) for c in candidates[:k]]
```

`hybrid.py`:

```python
class HybridRetriever:
    name = "hybrid"
    def __init__(self, legs: list[Retriever], fuser: Fuser, reranker: Reranker, *, k_leg: int, k_fused: int):
        ...; self.last_trace: dict[str, list[tuple[str, float]]] = {}
    def retrieve(self, query, k, filters=None): return self.retrieve_many([query], k, filters)
    def retrieve_many(self, queries, k, filters=None, rerank_query: str | None = None):
        self.last_trace = {}
        lists = []
        for q in queries:
            for leg in self.legs:
                hits = leg.retrieve(q, self.k_leg, filters)
                self.last_trace[f"{leg.name}:{q}"] = [(h.chunk.id, h.score) for h in hits]
                lists.append(hits)
        fused = self.fuser.fuse(lists, self.k_fused)
        self.last_trace["fused"] = [(h.chunk.id, h.score) for h in fused]
        reranked = self.reranker.rerank(rerank_query or queries[0], fused, k)
        self.last_trace["reranked"] = [(h.chunk.id, h.score) for h in reranked]
        return reranked
```

**Step 3: Run, expect 3 passed. Commit** (`feat: add RRF fusion, hybrid retriever, and noop reranker`).

---

### Task 16: Composition root, `scripts/build_index.py`, `scripts/search.py`

**Files:**
- Create: `src/legal_rag/factory.py`, `src/legal_rag/retrieval/index_builder.py`, `scripts/build_index.py`, `scripts/search.py`
- Test: `tests/legal_rag/integration/test_ingest_to_search.py`

**Step 1: Failing test**

```python
from pathlib import Path

from legal_rag.config import load_settings
from legal_rag.factory import build_hybrid_retriever
from legal_rag.ingestion.pipeline import run_ingestion
from legal_rag.retrieval.index_builder import build_index

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "raw"


def _settings(tmp_path):
    return load_settings(profile="default", overrides={
        "paths": {"root": str(tmp_path), "raw_dir": str(FIXTURES)}, "corpus": {"version": "it"},
        "embedding": {"backend": "fake"}, "retrieval": {"k_final": 3},
    })


def test_ingest_build_and_search(tmp_path):
    settings = _settings(tmp_path)
    run_ingestion(settings)
    report = build_index(settings)
    assert report.chunks > 5 and report.collection == "legal_it_fake_64"
    retriever = build_hybrid_retriever(settings)
    hits = retriever.retrieve("임차권등기명령 신청서에 적어야 하는 사항", k=3)
    assert any(h.chunk.path == "art:제3조의3/para:제2항" for h in hits)
    hits = retriever.retrieve("민법 제750조", k=3)
    assert hits[0].chunk.metadata.article == "제750조"
```

**Step 2: Implement**

`factory.py`:

```python
def build_embedder(settings) -> Embedder
def build_tokenizer(settings) -> Tokenizer
def collection_name(settings, embedder) -> str: return f"legal_{settings.corpus.version}_{slug(embedder.name)}"
def open_store(settings, embedder, *, recreate=False) -> VectorStore   # embedded or server by settings.qdrant.mode
def build_reranker(settings) -> Reranker                                # noop, or cross_encoder (import lazily; raise RuntimeError with install hint)
def build_hybrid_retriever(settings) -> HybridRetriever                 # loads BM25Stats from index_corpus_dir()/bm25.json
def build_chat_model(settings) -> ChatModel                             # Task 18
def build_orchestrator(settings) -> Orchestrator                        # Task 24
```

`index_builder.py`:

```python
@dataclass(frozen=True)
class IndexReport: chunks: int; collection: str; dimension: int; index_dir: Path

def build_index(settings) -> IndexReport:
    corpus = Corpus.load(settings.processed_corpus_dir())
    chunks = list(corpus.chunks())
    embedder, tok = build_embedder(settings), build_tokenizer(settings)
    stats = BM25Stats.fit({c.id: tok.tokenize(c.text) for c in chunks}, k1=settings.sparse.k1, b=settings.sparse.b)
    settings.index_corpus_dir().mkdir(parents=True, exist_ok=True)
    stats.save(settings.index_corpus_dir() / "bm25.json")
    store = open_store(settings, embedder, recreate=True)
    for batch in batched(chunks, settings.embedding.batch_size * 4):
        store.upsert(batch, dense=embedder.embed_documents([c.text for c in batch]),
                     sparse=[stats.document_vector(tok.tokenize(c.text)) for c in batch])
    (settings.index_corpus_dir() / "index.json").write_text(json.dumps({"collection": ..., "embedder": embedder.name, "dimension": ..., "tokenizer": tok.name, "chunks": len(chunks), "config_hash": settings.hash()}, ensure_ascii=False, indent=2))
    return IndexReport(len(chunks), collection_name(settings, embedder), embedder.dimension, settings.index_corpus_dir())
```

Scripts: `build_index.py` mirrors `ingest.py` options and prints `chunks=... collection=...`. `search.py` takes `QUERY`, `--k`, `--document-type`, `--json` and prints rank, score, chunk id, path, header line of text.

**Step 3: Run, expect 1 passed. Commit** (`feat: add the composition root, index builder, and search CLI`).

---

### Task 17: Retrieval evaluation

**Files:**
- Create: `src/legal_rag/evaluation/datasets.py`, `retrieval_metrics.py`, `retrieval_runner.py`, `scripts/eval_retrieval.py`
- Test: `tests/legal_rag/unit/test_retrieval_metrics.py`, `tests/legal_rag/integration/test_eval_retrieval.py`

**Step 1: Failing metric tests** (hand-computed)

```python
import math
from legal_rag.evaluation.retrieval_metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k

RANKED = ["a", "b", "c", "d"]; REL = {"b", "d", "z"}

def test_recall_and_precision():
    assert recall_at_k(RANKED, REL, 2) == 1 / 3
    assert recall_at_k(RANKED, REL, 4) == 2 / 3
    assert precision_at_k(RANKED, REL, 4) == 0.5

def test_mrr():
    assert mrr(RANKED, REL) == 0.5
    assert mrr(["x"], REL) == 0.0

def test_ndcg():
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert abs(ndcg_at_k(RANKED, REL, 4) - dcg / idcg) < 1e-9
```

**Step 2: Implement** the four pure functions; `ndcg` uses binary gains and an ideal ranking of `min(len(rel), k)` ones.

`datasets.py`: `load_retrieval_examples(path) -> list[RetrievalExample]`, `load_generation_examples(path)`.

`retrieval_runner.py`:

```python
class RetrievalRow(BaseModel): id: str; query: str; ranked_document_ids: list[str]; recall: dict[int, float]; precision: dict[int, float]; mrr: float; ndcg: dict[int, float]
class RetrievalReport(BaseModel): retriever: str; rows: list[RetrievalRow]; aggregates: dict[str, float]; def table(self) -> str

def run_retrieval_eval(retriever, examples, ks=(5, 10, 20)) -> RetrievalReport
```
Ranked ids are document ids deduplicated in order from chunk hits (`k = max(ks) * 3` chunks fetched, filters from the example).

Integration test: ingest + build (fake embedder) + run on `fixtures/eval/retrieval.jsonl`; assert `aggregates["recall@5"] >= 0.5` and `report.table()` contains `recall@5`. `scripts/eval_retrieval.py --dataset PATH --json OUT` prints the table.

**Step 3: Run, expect green. Commit** (`feat: add retrieval metrics, evaluation runner, and CLI`).

V0.2 exit criterion met.

---

## V0.3 — RAG chatbot baseline

### Task 18: ChatModel Protocol with fake and Ollama backends, prompt files

**Files:**
- Create: `src/legal_rag/llm/base.py`, `fake.py`, `ollama.py`, `prompts/__init__.py`, `prompts/analyze_system.md`, `prompts/rewrite_system.md`, `prompts/answer_system.md`
- Test: `tests/legal_rag/unit/test_llm.py`

**Step 1: Failing tests**

```python
import httpx, json
import pytest
from pydantic import BaseModel

from legal_rag.llm import FakeChatModel, Message, OllamaChatModel, StructuredOutputError
from legal_rag.llm.prompts import load_prompt


class Out(BaseModel):
    answer: str


def test_fake_model_returns_scripted_structured_responses_in_order():
    m = FakeChatModel({"Out": [Out(answer="first"), Out(answer="second")], "text": ["hi"]})
    assert m.generate([Message(role="user", content="x")], response_schema=Out).parsed.answer == "first"
    assert m.generate([Message(role="user", content="x")], response_schema=Out).parsed.answer == "second"
    assert m.generate([Message(role="user", content="x")]).text == "hi"
    assert len(m.calls) == 3 and m.calls[0].response_schema is Out


def test_ollama_model_sends_schema_and_parses(monkeypatch):
    seen = {}
    def handler(req: httpx.Request):
        seen.update(json.loads(req.read()))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": '{"answer": "ok"}'}})
    m = OllamaChatModel(model="qwen3:4b", base_url="http://t", transport=httpx.MockTransport(handler))
    r = m.generate([Message(role="user", content="q")], response_schema=Out)
    assert r.parsed.answer == "ok" and seen["format"]["type"] == "object" and seen["think"] is False and seen["stream"] is False


def test_ollama_invalid_json_raises_structured_output_error():
    handler = lambda req: httpx.Response(200, json={"message": {"content": "not json"}})
    m = OllamaChatModel(model="m", base_url="http://t", transport=httpx.MockTransport(handler))
    with pytest.raises(StructuredOutputError):
        m.generate([Message(role="user", content="q")], response_schema=Out)


def test_prompts_load():
    assert "LEGAL SOURCE" in load_prompt("answer_system")
```

**Step 2: Implement**

`base.py`:

```python
class Message(BaseModel): role: Literal["system", "user", "assistant"]; content: str
class ModelResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    text: str; parsed: BaseModel | None = None; raw: dict = {}
class StructuredOutputError(RuntimeError):
    def __init__(self, message, text): super().__init__(message); self.text = text
class ChatModel(Protocol): ...   # as in design
```

`fake.py` records `calls: list[FakeCall(messages, response_schema, temperature)]`; keys are `response_schema.__name__` or `"text"`; raises `AssertionError("no scripted response for X")` when exhausted.

`ollama.py`: POST `/api/chat` with `{"model", "messages", "stream": False, "think": False, "options": {"temperature", "num_predict"}, "format": schema.model_json_schema() if schema else None}` (omit `format` when None). Parse `message.content`; strip a leading ```` ```json ```` fence if present; `schema.model_validate_json` → on `ValidationError`/`JSONDecodeError` raise `StructuredOutputError`.

Prompt files (Korean instructions, English field names). `answer_system.md` must contain: role (한국 법률 정보 보조), the rule that only `[LEGAL SOURCE Sn]` ids may be cited, never invent 조문/사건번호, distinguish 확인된 법률 정보 vs 해석, ask for facts when missing, output must follow the JSON schema, respond in Korean. `analyze_system.md` asks for the `QueryAnalysis` JSON. `rewrite_system.md` asks for up to two 법률 검색 질의.

`prompts/__init__.py`: `load_prompt(name) -> str` via `importlib.resources.files(__package__).joinpath(f"{name}.md").read_text(encoding="utf-8")`. Add `[tool.hatch.build.targets.wheel] include` not needed for editable installs; ensure `prompts/*.md` are inside the package directory.

**Step 3: Run, expect 4 passed. Commit** (`feat: add chat model protocol with fake and Ollama backends and prompt files`).

---

### Task 19: Citation parser, verifier, and policy

**Files:**
- Create: `src/legal_rag/verification/citations.py`, `verifier.py`, `policy.py`
- Test: `tests/legal_rag/unit/test_verification.py`

**Step 1: Failing tests**

```python
from legal_rag.schemas import Citation, CitationKind, Confidence, DocumentType, Evidence, EvidenceBundle, LegalAnswer
from legal_rag.verification import VerificationAction, VerificationPolicy, Verifier


def _bundle():
    return EvidenceBundle(items=[
        Evidence(source_id="S1", chunk_id="c1", document_id="d1", document_type=DocumentType.statute,
                 header={"Type": "Statute", "Law": "주택임대차보호법", "Article": "제3조의3"}, text="임대차가 끝난 후 ... 임차권등기명령을 신청할 수 있다.", score=1.0),
        Evidence(source_id="S2", chunk_id="c2", document_id="d2", document_type=DocumentType.court_decision,
                 header={"Type": "Case", "Court": "서울중앙지방법원", "Case number": "2019가단12345"}, text="민법 제750조에 따른 ...", score=0.9),
    ])


def _answer(**kw):
    base = dict(summary="보증금을 돌려받을 수 있습니다.", legal_issues=["보증금 반환"], analysis="주택임대차보호법 제3조의3에 따라 임차권등기명령을 신청할 수 있습니다.",
                possible_actions=["임차권등기명령 신청"], additional_information_needed=[], citations=[Citation(source_id="S1", kind=CitationKind.statute, reference="주택임대차보호법 제3조의3")],
                uncertainty="", confidence=Confidence.high)
    base.update(kw); return LegalAnswer(**base)


def test_grounded_answer_passes():
    report = Verifier().verify(_answer(), _bundle())
    assert report.ok and report.issues == []


def test_unknown_source_id_and_foreign_statute_are_flagged():
    ans = _answer(citations=[Citation(source_id="S9", kind=CitationKind.statute, reference="민법 제750조")],
                  analysis="민법 제839조의2와 대법원 2015다99999 판결에 따르면 ...")
    report = Verifier().verify(ans, _bundle())
    kinds = {i.kind for i in report.issues}
    assert kinds == {"unknown_source_id", "statute_not_in_evidence", "case_not_in_evidence"}
    assert "민법 제839조의2" in " ".join(i.detail for i in report.issues)


def test_source_id_mismatch_when_reference_not_in_cited_block():
    ans = _answer(citations=[Citation(source_id="S1", kind=CitationKind.statute, reference="민법 제750조")])
    report = Verifier().verify(ans, _bundle())
    assert [i.kind for i in report.issues] == ["source_id_mismatch"]


def test_statute_mentioned_inside_evidence_text_counts_as_supported():
    ans = _answer(analysis="민법 제750조에 따라 손해배상 책임이 있습니다.", citations=[Citation(source_id="S2", kind=CitationKind.statute, reference="민법 제750조")])
    assert Verifier().verify(ans, _bundle()).ok


def test_policy_regenerates_then_strips_then_rejects():
    ans = _answer(citations=[Citation(source_id="S9", kind=CitationKind.statute, reference="민법 제839조의2")], analysis="민법 제839조의2에 따라 재산분할을 청구할 수 있습니다. 또한 보증금은 반환되어야 합니다.")
    report = Verifier().verify(ans, _bundle())
    policy = VerificationPolicy(max_regenerations=1)
    assert policy.decide(report, attempt=0) == VerificationAction.regenerate
    assert policy.decide(report, attempt=1) == VerificationAction.strip
    stripped = policy.strip(ans, report)
    assert stripped.citations == [] and "제839조의2" not in stripped.analysis and "보증금은 반환되어야 합니다." in stripped.analysis
    assert stripped.confidence == Confidence.low and "제839조의2" in stripped.uncertainty
    assert policy.decide_after_strip(stripped, evidence_required=True) == VerificationAction.reject
```

**Step 2: Implement**

`citations.py`: `extract_statute_refs`, `extract_case_refs` re-exported from `textutil.refs`, plus `answer_text(answer) -> str` joining summary, analysis, possible_actions, uncertainty.

`verifier.py`:

```python
class Verifier:
    def verify(self, answer, bundle) -> VerificationReport:
        by_id = {e.source_id: e for e in bundle.items}
        evidence_statutes = set(); evidence_cases = set()
        for e in bundle.items:
            law, art = e.header.get("Law"), e.header.get("Article")
            if art: evidence_statutes.add(f"{law} {art}" if law else art); evidence_statutes.add(art)
            if case := e.header.get("Case number"): evidence_cases.add(case)
            for r in extract_statute_refs(e.text): evidence_statutes.update({r.render(), r.article})
            evidence_cases.update(extract_case_refs(e.text))
        issues = []
        for c in answer.citations:
            if c.source_id not in by_id:
                issues.append(VerificationIssue(kind="unknown_source_id", detail=f"{c.source_id}: {c.reference}")); continue
            if not self._block_supports(by_id[c.source_id], c.reference):
                issues.append(VerificationIssue(kind="source_id_mismatch", detail=f"{c.source_id} does not contain {c.reference}"))
        text = answer_text(answer)
        checked_statutes = [r.render() for r in extract_statute_refs(text)]
        for ref in extract_statute_refs(text):
            if ref.render() not in evidence_statutes and ref.article not in evidence_statutes:
                issues.append(VerificationIssue(kind="statute_not_in_evidence", detail=ref.render()))
        checked_cases = extract_case_refs(text)
        for case in checked_cases:
            if case not in evidence_cases:
                issues.append(VerificationIssue(kind="case_not_in_evidence", detail=case))
        return VerificationReport(ok=not issues, issues=issues, checked_statutes=checked_statutes, checked_cases=checked_cases)

    def _block_supports(self, evidence, reference) -> bool:
        haystack = " ".join([*evidence.header.values(), evidence.text])
        refs = extract_statute_refs(reference)
        if refs:
            r = refs[0]
            in_header = evidence.header.get("Article") == r.article and (r.law_name is None or evidence.header.get("Law") == r.law_name)
            return in_header or any(x.article == r.article for x in extract_statute_refs(haystack))
        cases = extract_case_refs(reference)
        if cases:
            return cases[0] in extract_case_refs(haystack) or evidence.header.get("Case number") == cases[0]
        return reference.strip() in haystack
```

`policy.py`:

```python
class VerificationPolicy:
    def __init__(self, max_regenerations: int = 1): ...
    def decide(self, report, attempt) -> VerificationAction:
        if report.ok: return VerificationAction.accept
        return VerificationAction.regenerate if attempt < self.max_regenerations else VerificationAction.strip
    def strip(self, answer, report) -> LegalAnswer:
        bad_refs = {i.detail.split(": ", 1)[-1] for i in report.issues if i.kind in ("unknown_source_id", "source_id_mismatch")}
        bad_refs |= {i.detail for i in report.issues if i.kind in ("statute_not_in_evidence", "case_not_in_evidence")}
        bad_ids = {i.detail.split(":")[0] for i in report.issues if i.kind == "unknown_source_id"}
        citations = [c for c in answer.citations if c.source_id not in bad_ids and c.reference not in bad_refs and not any(b in c.reference for b in bad_refs)]
        def clean(text): return " ".join(s for s in split_sentences(text) if not any(b in s or _article_only(b) in s for b in bad_refs))
        note = "다음 인용은 제공된 근거에서 확인되지 않아 제거되었습니다: " + ", ".join(sorted(bad_refs))
        return answer.model_copy(update={"citations": citations, "analysis": clean(answer.analysis), "summary": clean(answer.summary),
                                         "possible_actions": [a for a in answer.possible_actions if not any(b in a for b in bad_refs)],
                                         "confidence": Confidence.low, "uncertainty": (answer.uncertainty + " " + note).strip()})
    def decide_after_strip(self, answer, *, evidence_required: bool) -> VerificationAction:
        return VerificationAction.reject if evidence_required and not answer.citations else VerificationAction.accept
    def feedback(self, report) -> str:   # message appended for regeneration
        return "이전 답변에서 근거 없는 인용이 발견되었습니다: " + "; ".join(f"{i.kind}: {i.detail}" for i in report.issues) + " 제공된 [LEGAL SOURCE] 자료만 인용해 다시 작성하세요."
```

`_article_only("민법 제839조의2")` returns `"제839조의2"` so a sentence citing the bare article is also removed.

**Step 3: Run, expect 5 passed. Commit** (`feat: add deterministic citation verification and handling policy`).

---

### Task 20: Query analyzer and clarification policy

**Files:**
- Create: `src/legal_rag/rag/analyzer.py`, `clarify.py`
- Test: `tests/legal_rag/unit/test_analyzer.py`

**Step 1: Failing tests**

```python
from legal_rag.llm import FakeChatModel
from legal_rag.rag.analyzer import QueryAnalyzer
from legal_rag.rag.clarify import ClarificationPolicy
from legal_rag.schemas import DocumentType, QueryAnalysis, Turn
from datetime import datetime


def test_analyzer_merges_deterministic_refs_over_model_output():
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis(domain="민사", subdomain="임대차", issues=["보증금 반환"], keywords=["보증금"], statute_refs=["엉뚱한 법 제1조"])]})
    analysis = QueryAnalyzer(model).analyze("주택임대차보호법 제3조의3에 따라 집주인이 보증금을 안 줍니다", history=[])
    assert analysis.statute_refs == ["주택임대차보호법 제3조의3"] and analysis.domain == "민사"
    sent = model.calls[0].messages
    assert sent[0].role == "system" and "집주인" in sent[-1].content


def test_analyzer_includes_recent_history_only():
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis()]})
    history = [Turn(role="user", content=f"turn {i}", created_at=datetime(2026, 1, 1)) for i in range(10)]
    QueryAnalyzer(model, history_turns=2).analyze("질문", history=history)
    content = model.calls[0].messages[-1].content
    assert "turn 9" in content and "turn 8" in content and "turn 7" not in content


def test_analyzer_falls_back_to_deterministic_analysis_on_model_failure():
    class Broken:
        name = "broken"
        def generate(self, *a, **k): raise RuntimeError("down")
    analysis = QueryAnalyzer(Broken()).analyze("민법 제750조 손해배상", history=[])
    assert analysis.statute_refs == ["민법 제750조"] and analysis.needs_more_info is False


def test_clarification_policy():
    p = ClarificationPolicy(allow=True)
    assert p.should_clarify(QueryAnalysis(needs_more_info=True, missing_facts=["계약 종료일"])) is True
    assert p.should_clarify(QueryAnalysis(needs_more_info=True, missing_facts=["x"], statute_refs=["민법 제750조"])) is False
    assert p.should_clarify(QueryAnalysis(needs_more_info=True)) is False
    assert ClarificationPolicy(allow=False).should_clarify(QueryAnalysis(needs_more_info=True, missing_facts=["x"])) is False
```

**Step 2: Implement**

`analyzer.py`:

```python
class QueryAnalyzer:
    def __init__(self, model: ChatModel, *, history_turns: int = 4): ...
    def analyze(self, message: str, history: list[Turn]) -> QueryAnalysis:
        statute_refs = [r.render() for r in extract_statute_refs(message)]
        case_refs = extract_case_refs(message)
        user_block = self._render(message, history[-self.history_turns:] if self.history_turns else [])
        try:
            resp = self.model.generate([Message(role="system", content=load_prompt("analyze_system")), Message(role="user", content=user_block)], response_schema=QueryAnalysis, temperature=0.0)
            base = resp.parsed
        except Exception:
            base = QueryAnalysis()
        return base.model_copy(update={"statute_refs": statute_refs or base.statute_refs, "case_refs": case_refs or base.case_refs})
```

`_render` formats `이전 대화:` lines then `현재 질문: ...`. (`QueryAnalysis` is frozen, `model_copy(update=...)` is fine.)

`clarify.py`:

```python
class ClarificationPolicy:
    def __init__(self, allow: bool = True): self.allow = allow
    def should_clarify(self, a: QueryAnalysis) -> bool:
        return self.allow and a.needs_more_info and bool(a.missing_facts) and not (a.statute_refs or a.case_refs)
```

**Step 3: Run, expect 4 passed. Commit** (`feat: add query analyzer with deterministic reference merging and clarification policy`).

---

### Task 21: Query rewriter and context builder

**Files:**
- Create: `src/legal_rag/rag/rewriter.py`, `context.py`
- Test: `tests/legal_rag/unit/test_rewriter_context.py`

**Step 1: Failing tests**

```python
from legal_rag.rag.context import ContextBuilder
from legal_rag.rag.rewriter import QueryRewriter
from legal_rag.schemas import QueryAnalysis, RewrittenQueries, RetrievalFilters, DocumentType, ScoredChunk
from tests.legal_rag.unit.test_store import _chunks


def test_deterministic_rewrite_adds_refs_and_keywords():
    a = QueryAnalysis(issues=["보증금 반환"], keywords=["임대차", "보증금"], statute_refs=["주택임대차보호법 제3조의3"], document_types_needed=[DocumentType.statute], legal_domain=None) if False else \
        QueryAnalysis(issues=["보증금 반환"], keywords=["임대차", "보증금"], statute_refs=["주택임대차보호법 제3조의3"], document_types_needed=[DocumentType.statute])
    rq = QueryRewriter(model=None).rewrite("집주인이 보증금을 안 줍니다", a)
    assert rq.search_queries[0] == "집주인이 보증금을 안 줍니다"
    assert "주택임대차보호법 제3조의3" in rq.search_queries
    assert any("보증금 반환" in q and "임대차" in q for q in rq.search_queries)
    assert len(rq.search_queries) <= 4


def test_filters_from_analysis():
    a = QueryAnalysis(domain="민사", document_types_needed=[DocumentType.court_decision])
    f = QueryRewriter(model=None).filters(a)
    assert f.document_types == [DocumentType.court_decision] and f.legal_domain == "민사"
    assert QueryRewriter(model=None).filters(QueryAnalysis()) is None


def test_context_builder_assigns_ids_and_headers():
    c1, c2, c3 = _chunks()
    hits = [ScoredChunk(chunk=c3, score=0.9, retriever="reranked"), ScoredChunk(chunk=c1, score=0.8, retriever="reranked")]
    bundle = ContextBuilder(max_chars=1500).build(hits)
    assert [e.source_id for e in bundle.items] == ["S1", "S2"]
    assert bundle.items[0].header["Type"] == "Case" and bundle.items[0].header["Case number"] == "2018다12345"
    assert bundle.items[1].header["Law"] == "주택임대차보호법" and bundle.items[1].header["Article"] == "제3조의2"
    assert "[LEGAL SOURCE S1]" in bundle.render()


def test_context_builder_truncates_long_text():
    c1, *_ = _chunks()
    long = c1.model_copy(update={"text": "가" * 5000})
    bundle = ContextBuilder(max_chars=100).build([ScoredChunk(chunk=long, score=1, retriever="reranked")])
    assert len(bundle.items[0].text) <= 101 and bundle.items[0].text.endswith("…")
```

(Clean the first test's odd conditional expression when writing it; keep the second form.)

**Step 2: Implement**

`rewriter.py`:

```python
class QueryRewriter:
    def __init__(self, model: ChatModel | None, *, max_queries: int = 4): ...
    def rewrite(self, message, analysis) -> RewrittenQueries:
        queries = [message]
        queries += [ref for ref in analysis.statute_refs]
        queries += analysis.case_refs
        if analysis.issues or analysis.keywords:
            queries.append(" ".join(dict.fromkeys([*analysis.issues, *analysis.keywords])))
        if self.model is not None:
            try:
                resp = self.model.generate([Message(role="system", content=load_prompt("rewrite_system")), Message(role="user", content=message)], response_schema=RewrittenQueries, temperature=0.0)
                queries += resp.parsed.search_queries
            except Exception:
                pass
        deduped = list(dict.fromkeys(q.strip() for q in queries if q.strip()))
        return RewrittenQueries(original=message, search_queries=deduped[: self.max_queries])

    def filters(self, analysis) -> RetrievalFilters | None:
        f = RetrievalFilters(document_types=analysis.document_types_needed or None, legal_domain=analysis.domain)
        return None if f == RetrievalFilters() else f
```

Note: filtering by `document_types_needed` can hide useful cross-type evidence; V0.3 applies only `legal_domain` and leaves `document_types` as a soft hint — implement `filters()` returning `RetrievalFilters(legal_domain=analysis.domain)` when domain is known and document types only when `analysis.document_types_needed` has exactly one entry. Update the test accordingly.

`context.py`:

```python
class ContextBuilder:
    def __init__(self, max_chars: int = 1500): ...
    def build(self, hits: list[ScoredChunk]) -> EvidenceBundle:
        items = []
        for n, hit in enumerate(hits, start=1):
            m = hit.chunk.metadata
            if hit.chunk.document_type == DocumentType.statute:
                header = {"Type": "Statute", "Law": m.law_name or hit.chunk.title, "Article": m.article or "", **({"Paragraph": m.paragraph} if m.paragraph else {}), **({"Effective date": str(m.effective_date)} if m.effective_date else {})}
            elif hit.chunk.document_type == DocumentType.court_decision:
                header = {"Type": "Case", "Court": m.court or "", "Case number": m.case_number or "", "Decision date": str(m.decision_date or ""), "Section": m.section or ""}
            else:
                header = {"Type": "Interpretation", "Authority": m.authority or "", "Reference number": m.reference_number or ""}
            header = {k: v for k, v in header.items() if v != ""}
            text = hit.chunk.text if len(hit.chunk.text) <= self.max_chars else hit.chunk.text[: self.max_chars] + "…"
            items.append(Evidence(source_id=f"S{n}", chunk_id=hit.chunk.id, document_id=hit.chunk.document_id, document_type=hit.chunk.document_type, header=header, text=text, score=hit.score))
        return EvidenceBundle(items=items)
```

**Step 3: Run, expect 4 passed. Commit** (`feat: add query rewriter and evidence context builder`).

---

### Task 22: Generator

**Files:**
- Create: `src/legal_rag/rag/generator.py`
- Test: `tests/legal_rag/unit/test_generator.py`

**Step 1: Failing tests**

```python
import pytest
from legal_rag.llm import FakeChatModel, StructuredOutputError
from legal_rag.rag.generator import Generator, GenerationError
from tests.legal_rag.unit.test_verification import _answer, _bundle


def test_generator_builds_messages_with_evidence_and_parses():
    model = FakeChatModel({"LegalAnswer": [_answer()]})
    out = Generator(model).generate("보증금을 못 받았어요", _bundle(), feedback=None)
    assert out.summary.startswith("보증금")
    msgs = model.calls[0].messages
    assert msgs[0].role == "system" and "[LEGAL SOURCE S1]" in msgs[1].content and msgs[-1].content.endswith("보증금을 못 받았어요")


def test_generator_retries_once_on_structured_output_error():
    class Flaky:
        name = "flaky"
        def __init__(self): self.n = 0
        def generate(self, messages, **kw):
            self.n += 1
            if self.n == 1: raise StructuredOutputError("bad", "garbage")
            from legal_rag.llm import ModelResponse
            return ModelResponse(text="{}", parsed=_answer())
    g = Generator(Flaky())
    assert g.generate("q", _bundle(), feedback=None).confidence
    assert g.model.n == 2


def test_generator_raises_after_retry_fails():
    class Dead:
        name = "dead"
        def generate(self, *a, **k): raise StructuredOutputError("bad", "x")
    with pytest.raises(GenerationError):
        Generator(Dead()).generate("q", _bundle(), feedback=None)


def test_feedback_is_appended_as_assistant_user_exchange():
    model = FakeChatModel({"LegalAnswer": [_answer()]})
    Generator(model).generate("q", _bundle(), feedback="근거 없는 인용")
    assert any(m.role == "user" and "근거 없는 인용" in m.content for m in model.calls[0].messages)
```

**Step 2: Implement**

```python
class GenerationError(RuntimeError): ...

class Generator:
    def __init__(self, model: ChatModel, *, temperature: float = 0.2, max_tokens: int = 1024): ...
    def generate(self, message, bundle, *, feedback: str | None) -> LegalAnswer:
        messages = [Message(role="system", content=load_prompt("answer_system")),
                    Message(role="user", content="다음은 검색된 법률 근거입니다.\n\n" + bundle.render()),
                    Message(role="user", content="질문: " + message)]
        if feedback:
            messages.append(Message(role="user", content=feedback))
        last_error = None
        for attempt in range(2):
            try:
                resp = self.model.generate(messages, response_schema=LegalAnswer, temperature=self.temperature, max_tokens=self.max_tokens)
                return resp.parsed
            except StructuredOutputError as exc:
                last_error = exc
                messages = messages + [Message(role="user", content=f"출력이 JSON 스키마에 맞지 않았습니다: {exc}. 스키마에 맞는 JSON만 출력하세요.")]
        raise GenerationError(str(last_error))
```

Adjust the first test's final assertion to match: the last message content is `"질문: 보증금을 못 받았어요"`.

**Step 3: Run, expect 4 passed. Commit** (`feat: add the structured answer generator with a repair retry`).

---

### Task 23: Conversation store and trace

**Files:**
- Create: `src/legal_rag/rag/conversation.py`, `trace.py`
- Test: `tests/legal_rag/unit/test_conversation_trace.py`

**Step 1: Failing tests**

```python
import json
from datetime import datetime
from legal_rag.rag.conversation import InMemoryConversationStore, SQLiteConversationStore
from legal_rag.rag.trace import TraceRecorder
from legal_rag.schemas import Turn


def test_in_memory_store_appends_and_creates_on_get():
    s = InMemoryConversationStore()
    assert s.get("c1").turns == []
    s.append("c1", Turn(role="user", content="hi", created_at=datetime(2026, 1, 1)))
    assert [t.content for t in s.get("c1").turns] == ["hi"]


def test_sqlite_store_persists(tmp_path):
    path = tmp_path / "conv.db"
    SQLiteConversationStore(path).append("c1", Turn(role="assistant", content="a", created_at=datetime(2026, 1, 1), evidence_ids=["x#y"]))
    turns = SQLiteConversationStore(path).get("c1").turns
    assert turns[0].evidence_ids == ["x#y"]


def test_trace_recorder_times_stages_and_writes_json_line(tmp_path):
    rec = TraceRecorder(conversation_id="c1", user_text="비밀", include_text=False, path=tmp_path / "trace.jsonl")
    with rec.stage("analyze"):
        pass
    rec.trace.search_queries = ["q"]
    rec.finish(policy_action="accept")
    line = json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert line["timings"][0]["stage"] == "analyze" and line["user_text"] is None and line["policy_action"] == "accept"
```

**Step 2: Implement** both stores behind the `ConversationStore` Protocol; SQLite table `turns(conversation_id TEXT, seq INTEGER, payload TEXT)`. `TraceRecorder` holds a mutable `Trace`, `stage(name)` is a context manager using `time.perf_counter`, `finish(...)` sets fields and appends `trace.model_dump_json()` to `path` if given.

**Step 3: Run, expect 3 passed. Commit** (`feat: add conversation stores and a per-turn trace recorder`).

---

### Task 24: Orchestrator and `scripts/chat.py`

**Files:**
- Create: `src/legal_rag/rag/orchestrator.py`, `scripts/chat.py`
- Modify: `src/legal_rag/factory.py` (`build_chat_model`, `build_orchestrator`)
- Test: `tests/legal_rag/integration/test_chat_turn.py`

**Step 1: Failing tests** (fake embedder, fake model, embedded Qdrant on fixtures)

```python
from pathlib import Path
import pytest

from legal_rag.config import load_settings
from legal_rag.factory import build_hybrid_retriever
from legal_rag.ingestion.pipeline import run_ingestion
from legal_rag.llm import FakeChatModel
from legal_rag.rag.orchestrator import Orchestrator
from legal_rag.rag.conversation import InMemoryConversationStore
from legal_rag.retrieval.index_builder import build_index
from legal_rag.schemas import Citation, CitationKind, Confidence, LegalAnswer, QueryAnalysis
from legal_rag.verification import VerificationPolicy, Verifier

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "raw"


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("idx")
    settings = load_settings(profile="default", overrides={"paths": {"root": str(tmp), "raw_dir": str(FIXTURES)}, "corpus": {"version": "chat"}, "embedding": {"backend": "fake"}})
    run_ingestion(settings); build_index(settings)
    return settings, build_hybrid_retriever(settings)


def _answer(source_id="S1", reference="주택임대차보호법 제3조의3", analysis="주택임대차보호법 제3조의3에 따라 임차권등기명령을 신청할 수 있습니다."):
    return LegalAnswer(summary="보증금 반환을 청구할 수 있습니다.", legal_issues=["보증금 반환"], analysis=analysis, possible_actions=["임차권등기명령 신청"],
                       additional_information_needed=[], citations=[Citation(source_id=source_id, kind=CitationKind.statute, reference=reference)], uncertainty="", confidence=Confidence.medium)


def _orchestrator(retriever, model, settings):
    return Orchestrator.from_components(settings=settings, retriever=retriever, model=model, store=InMemoryConversationStore(), verifier=Verifier(), policy=VerificationPolicy(max_regenerations=1))


def test_grounded_turn_returns_answer_with_evidence(retriever):
    settings, r = retriever
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis(domain="민사", issues=["보증금 반환"], keywords=["임차권등기명령"])], "LegalAnswer": [_answer()]})
    resp = _orchestrator(r, model, settings).handle("임차권등기명령 신청서에 적어야 하는 사항은?", conversation_id="c1")
    assert resp.kind == "answer" and resp.verification.ok
    assert any(e.header.get("Article") == "제3조의3" for e in resp.evidence.items)
    assert resp.trace.timings and resp.trace.policy_action == "accept"


def test_hallucinated_citation_triggers_regeneration_then_strip(retriever):
    settings, r = retriever
    bad = _answer(source_id="S9", reference="민법 제839조의2", analysis="민법 제839조의2에 따라 청구할 수 있습니다. 보증금은 반환되어야 합니다.")
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis(issues=["보증금 반환"])], "LegalAnswer": [bad, bad]})
    resp = _orchestrator(r, model, settings).handle("보증금을 못 받았어요", conversation_id="c2")
    assert resp.kind == "answer" and resp.trace.regenerations == 1 and resp.trace.policy_action == "strip"
    assert resp.answer.confidence == Confidence.low and resp.answer.citations == []
    assert len([c for c in model.calls if c.response_schema.__name__ == "LegalAnswer"]) == 2


def test_missing_facts_returns_clarification_without_retrieval(retriever):
    settings, r = retriever
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis(needs_more_info=True, missing_facts=["임대차 계약 종료일", "보증금 액수"])]})
    resp = _orchestrator(r, model, settings).handle("법적으로 어떻게 되나요?", conversation_id="c3")
    assert resp.kind == "clarification" and resp.questions == ["임대차 계약 종료일", "보증금 액수"] and resp.evidence is None


def test_history_is_stored_but_evidence_text_is_not(retriever):
    settings, r = retriever
    store = InMemoryConversationStore()
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis()], "LegalAnswer": [_answer()]})
    orch = Orchestrator.from_components(settings=settings, retriever=r, model=model, store=store, verifier=Verifier(), policy=VerificationPolicy())
    orch.handle("질문", conversation_id="c4")
    turns = store.get("c4").turns
    assert [t.role for t in turns] == ["user", "assistant"] and turns[1].evidence_ids and "LEGAL SOURCE" not in turns[1].content
```

**Step 2: Implement `orchestrator.py`**

```python
class Orchestrator:
    def __init__(self, *, analyzer, clarifier, rewriter, retriever, context_builder, generator, verifier, policy, store, settings): ...

    @classmethod
    def from_components(cls, *, settings, retriever, model, store, verifier, policy):
        return cls(analyzer=QueryAnalyzer(model, history_turns=settings.rag.history_turns), clarifier=ClarificationPolicy(settings.rag.allow_clarification),
                   rewriter=QueryRewriter(model if settings.rag.llm_rewrite else None), retriever=retriever, context_builder=ContextBuilder(settings.rag.max_evidence_chars),
                   generator=Generator(model, temperature=settings.llm.temperature, max_tokens=settings.llm.max_tokens), verifier=verifier, policy=policy, store=store, settings=settings)

    def handle(self, message: str, *, conversation_id: str) -> ChatResponse:
        rec = TraceRecorder(conversation_id=conversation_id, user_text=message, include_text=self.settings.trace.include_text, path=self.settings.trace.path, model=getattr(self.generator.model, "name", None))
        history = self.store.get(conversation_id).turns
        self.store.append(conversation_id, Turn(role="user", content=message, created_at=_now()))
        with rec.stage("analyze"):
            analysis = self.analyzer.analyze(message, history)
        rec.trace.analysis = analysis
        if self.clarifier.should_clarify(analysis):
            rec.finish(policy_action="clarify")
            self._remember(conversation_id, "assistant", "\n".join(analysis.missing_facts), [])
            return ChatResponse(kind="clarification", conversation_id=conversation_id, questions=analysis.missing_facts, trace=rec.trace)
        with rec.stage("rewrite"):
            queries = self.rewriter.rewrite(message, analysis); filters = self.rewriter.filters(analysis)
        rec.trace.search_queries = queries.search_queries
        with rec.stage("retrieve"):
            hits = self.retriever.retrieve_many(queries.search_queries, self.settings.retrieval.k_final, filters, rerank_query=message) if hasattr(self.retriever, "retrieve_many") else self.retriever.retrieve(message, self.settings.retrieval.k_final, filters)
        rec.trace.retrieved = dict(getattr(self.retriever, "last_trace", {}))
        with rec.stage("context"):
            bundle = self.context_builder.build(hits)
        rec.trace.evidence_ids = [e.chunk_id for e in bundle.items]
        feedback = None; answer = None; report = None; action = None
        for attempt in range(self.settings.rag.max_regenerations + 1):
            with rec.stage(f"generate[{attempt}]"):
                answer = self.generator.generate(message, bundle, feedback=feedback)
            with rec.stage(f"verify[{attempt}]"):
                report = self.verifier.verify(answer, bundle)
            action = self.policy.decide(report, attempt)
            if action != VerificationAction.regenerate:
                break
            rec.trace.regenerations += 1
            feedback = self.policy.feedback(report)
        if action == VerificationAction.strip:
            answer = self.policy.strip(answer, report)
            action = self.policy.decide_after_strip(answer, evidence_required=bool(bundle.items)) if False else VerificationAction.strip
            # Evidence was found but nothing survived: reject only when the stripped answer has no citations AND no analysis text.
            if not answer.citations and not answer.analysis.strip():
                action = VerificationAction.reject
        rec.trace.verification = report
        rec.finish(policy_action=action.value)
        if action == VerificationAction.reject:
            self._remember(conversation_id, "assistant", "근거를 확인할 수 없어 답변을 제공하지 못했습니다.", rec.trace.evidence_ids)
            return ChatResponse(kind="rejected", conversation_id=conversation_id, reason="unsupported citations", evidence=bundle, verification=report, trace=rec.trace)
        self._remember(conversation_id, "assistant", answer.summary + "\n" + answer.analysis, rec.trace.evidence_ids)
        return ChatResponse(kind="answer", conversation_id=conversation_id, answer=answer, evidence=bundle, verification=report, trace=rec.trace)
```

Replace the awkward `if False else` line with the intended logic: after `strip`, set `action = VerificationAction.reject` when `not answer.citations and not answer.analysis.strip()`, else keep `strip`. `_remember` appends an assistant `Turn`. Keep `policy_action` values `accept | regenerate | strip | reject | clarify`.

`factory.build_chat_model`: `ollama` → `OllamaChatModel(model, base_url, timeout)`, `fake` → `FakeChatModel({})`, `transformers` → `RuntimeError("available in V0.4")`. `build_orchestrator(settings)` wires `SQLiteConversationStore(settings.index_corpus_dir() / "conversations.db")`.

`scripts/chat.py`: Typer command with `--profile`, `--conversation-id` (default random hex), `--trace/--no-trace`. Loop on `input("> ")`; on `ChatResponse`: print `요약`, `쟁점`, `분석`, `가능한 조치`, `추가로 필요한 정보`, `인용` (`S1 · 주택임대차보호법 제3조의3`), `불확실성`, `신뢰도`; for clarification print the questions; for rejected print the reason. `/sources` prints the last evidence bundle; `/quit` exits. With `--trace`, print stage timings and policy action after each turn.

**Step 3: Run all tests, expect green. Commit** (`feat: add the chat orchestrator and CLI`).

V0.3 CLI milestone: `uv run python scripts/ingest.py --raw-dir <aihub dir>`, `uv run python scripts/build_index.py`, `uv run python scripts/chat.py` against Ollama.

---

### Task 25: FastAPI application

**Files:**
- Create: `src/legal_rag/api/models.py`, `deps.py`, `routes.py`, `app.py`
- Test: `tests/legal_rag/integration/test_api.py`

**Step 1: Failing tests**

```python
from fastapi.testclient import TestClient
from legal_rag.api.app import create_app
from legal_rag.llm import FakeChatModel
from legal_rag.rag.conversation import InMemoryConversationStore
from legal_rag.rag.orchestrator import Orchestrator
from legal_rag.schemas import QueryAnalysis
from legal_rag.verification import VerificationPolicy, Verifier
from tests.legal_rag.integration.test_chat_turn import _answer, retriever  # noqa: F401


def _client(retriever, model):
    settings, r = retriever
    orch = Orchestrator.from_components(settings=settings, retriever=r, model=model, store=InMemoryConversationStore(), verifier=Verifier(), policy=VerificationPolicy())
    return TestClient(create_app(settings=settings, orchestrator=orch, retriever=r))


def test_health(retriever):
    c = _client(retriever, FakeChatModel({}))
    body = c.get("/health").json()
    assert body["status"] == "ok" and body["collection"].startswith("legal_chat_") and body["llm"] == "fake"


def test_search_endpoint(retriever):
    c = _client(retriever, FakeChatModel({}))
    body = c.post("/v1/search", json={"query": "민법 제750조", "k": 3}).json()
    assert body["hits"][0]["chunk"]["metadata"]["article"] == "제750조"


def test_chat_endpoint_round_trip(retriever):
    model = FakeChatModel({"QueryAnalysis": [QueryAnalysis()], "LegalAnswer": [_answer()]})
    c = _client(retriever, model)
    body = c.post("/v1/chat", json={"message": "보증금을 못 받았어요"}).json()
    assert body["kind"] == "answer" and body["conversation_id"] and body["answer"]["citations"][0]["source_id"] == "S1"
    assert "trace" not in body or body["trace"] is None


def test_chat_validation_error():
    from legal_rag.config import load_settings
    app = create_app(settings=load_settings(profile="default", overrides={"embedding": {"backend": "fake"}, "llm": {"backend": "fake"}}), orchestrator=None, retriever=None)
    assert TestClient(app).post("/v1/chat", json={}).status_code == 422
```

**Step 2: Implement**

`models.py`: `ChatRequest(message: str = Field(min_length=1, max_length=4000), conversation_id: str | None = None, include_trace: bool = False)`, `SearchRequest(query, k: int = Field(5, ge=1, le=50), filters: RetrievalFilters | None = None)`, `SearchResponse(hits: list[ScoredChunk])`, `HealthResponse(status, collection, chunks, embedder, llm, tokenizer)`.

`app.py`:

```python
def create_app(settings=None, orchestrator=None, retriever=None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="Korean Legal RAG", version=__version__)
    app.state.settings = settings
    app.state.orchestrator = orchestrator; app.state.retriever = retriever
    @app.on_event("startup")  # or lifespan
    def _startup():
        if app.state.retriever is None: app.state.retriever = build_hybrid_retriever(settings)
        if app.state.orchestrator is None: app.state.orchestrator = build_orchestrator(settings)
    app.include_router(router)
    return app

app = create_app()   # for `uvicorn legal_rag.api.app:app`
```

Use a lifespan context instead of the deprecated `on_event`. For the validation-error test with `orchestrator=None` and `retriever=None`, lazy construction must only happen on first use, not at import, and the `/v1/chat` 422 must be produced before any dependency runs (FastAPI validates the body before calling the handler, so this holds as long as dependencies are resolved lazily inside the handler).

`routes.py`: `/health` reads `index.json` from `settings.index_corpus_dir()` for `collection`/`chunks`; `/v1/search` calls `retriever.retrieve(query, k, filters)`; `/v1/chat` runs `await run_in_threadpool(orchestrator.handle, message, conversation_id=cid or uuid4().hex)` and drops `trace` unless `include_trace`.

`scripts/serve.py`: `uvicorn.run("legal_rag.api.app:app", host=settings.api.host, port=settings.api.port)`.

**Step 3: Run all tests, expect green. Commit** (`feat: add the FastAPI chat and search endpoints`).

---

### Task 26: Live tests and documentation

**Files:**
- Create: `tests/legal_rag/live/test_ollama_live.py` (skipped unless `LEGAL_RAG_LIVE=1`; checks `OllamaEmbedder("bge-m3").dimension == 1024` and a real `OllamaChatModel` returns a valid `LegalAnswer` for a fixture question)
- Create: `docs/legal-rag-backend.md` — how to install, pull `bge-m3`, ingest, build, search, evaluate, chat, serve; the config keys; how to add a new AI Hub layout (detect fingerprint + normalizer + fixture)
- Modify: `README.md` — add a section pointing at `docs/legal-rag-backend.md`
- Modify: `docs/integration-roadmap.md` — add a row for the legal RAG backend with status per version

**Commit** (`docs: document the legal RAG backend workflow`).

---

## Final verification

1. `uv run pytest tests/legal_rag -q` → all green, no network.
2. `uv run python scripts/ingest.py --raw-dir tests/legal_rag/fixtures/raw --corpus-version demo` → `documents=5`.
3. `LEGAL_RAG__EMBEDDING__BACKEND=fake uv run python scripts/build_index.py --corpus-version demo` → `collection=legal_demo_fake_64`.
4. `LEGAL_RAG__EMBEDDING__BACKEND=fake uv run python scripts/search.py "민법 제750조" --corpus-version demo` → top hit article `제750조`.
5. `LEGAL_RAG__EMBEDDING__BACKEND=fake uv run python scripts/eval_retrieval.py --dataset tests/legal_rag/fixtures/eval/retrieval.jsonl --corpus-version demo` → table with recall@5.
6. With Ollama running and `ollama pull bge-m3` done: `uv run python scripts/build_index.py --corpus-version demo` then `uv run python scripts/chat.py --corpus-version demo --trace`, ask `집주인이 보증금을 돌려주지 않습니다`, confirm the answer cites `S`-ids and verification accepts or strips with a trace line.
7. `LEGAL_RAG_LIVE=1 uv run pytest tests/legal_rag/live -q` on the host.

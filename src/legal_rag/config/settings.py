"""Typed settings for the legal RAG backend.

Precedence, highest first: environment variables (``LEGAL_RAG__SECTION__KEY``),
YAML profile values passed as init kwargs by :func:`legal_rag.config.load_settings`,
then the defaults declared here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    timeout_seconds: float = 120.0
    # Ollama context window for query embeddings; None keeps Ollama's default. Must match
    # the chat app's OLLAMA_EMBED_NUM_CTX or Ollama reloads bge-m3 between the two clients.
    query_num_ctx: int | None = Field(default=None, ge=1)


class SparseSettings(_Section):
    tokenizer: Literal["regex", "kiwi"] = "regex"
    k1: float = 1.2
    b: float = 0.75


class QdrantSettings(_Section):
    mode: Literal["embedded", "server"] = "embedded"
    path: Path | None = None  # defaults to <index_dir>/<corpus_version>/qdrant
    url: str = "http://localhost:6333"
    api_key: SecretStr | None = None  # environment only, never YAML
    # qdrant-client defaults to 5 seconds. Upserting a large corpus keeps the server busy
    # building payload indexes, and a 5-second write timeout kills the build mid-run.
    timeout_seconds: float = 120.0


class RetrievalSettings(_Section):
    k_dense: int = 20
    k_sparse: int = 20
    k_fused: int = 20
    k_final: int = 5
    rrf_k: int = 60
    corpora: list[str] = Field(default_factory=list)  # empty means query corpus.version alone
    corpus_weights: dict[str, float] = Field(default_factory=dict)  # absent corpora weigh 1.0
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    # Top hits of every leg that always reach the reranker, whatever their fused score. 0 is plain
    # RRF; see RRFFuser and docs/retrieval-diagnostics.md for why fan-out needs it.
    leg_floor: int = Field(default=0, ge=0)


class RerankSettings(_Section):
    backend: Literal["noop", "cross_encoder"] = "noop"
    model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cpu"
    # float16 needs a GPU. On the 4060 it reranks 50 candidates in 0.8 s instead of 3.0 s
    # with 1.2 GB instead of 2.5 GB, and the same top 10 (docs/legal-rag-backend.md).
    dtype: Literal["float32", "float16"] = "float32"
    # Load from the local Hugging Face cache without Hub checks: 1.9 s instead of 5.6 s.
    # Needs the model downloaded once; a fresh machine keeps the default.
    local_files_only: bool = False


class LLMSettings(_Section):
    backend: Literal["ollama", "fake", "transformers", "gemini"] = "ollama"
    model: str = "qwen3:4b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 1024
    adapter: str | None = None
    api_key: SecretStr | None = None  # Gemini key; environment/.env only
    gemini_thinking_level: Literal["low", "medium", "high"] | None = None
    timeout_seconds: float = 120.0
    # Ollama context window; None keeps Ollama's default (4096 for qwen3:4b here), which
    # silently cut the ~6,100-token answer prompts. Covers the generated tokens too.
    num_ctx: int | None = Field(default=None, ge=1)


class RagSettings(_Section):
    allow_clarification: bool = True
    llm_rewrite: bool = False
    # Turn the analyzer's single needed document kind into a hard retrieval filter. Off: on
    # 210 eval questions it asked for statutes 201 times, and filtering cut gold Success@10
    # from 0.557 to 0.414 (95% CI of the change [-0.205, -0.086], 2026-09-27).
    hard_filter: bool = False
    history_turns: int = 4
    max_regenerations: int = 1
    max_evidence_chars: int = 1500


class ApiSettings(_Section):
    host: str = "127.0.0.1"
    port: int = 8000
    # Run one search in the background at startup so BM25, the embedder and the
    # reranker load before the first user request (~18 s cold, 2.2 s warm).
    warmup: bool = False


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
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        # Environment beats YAML (which arrives as init kwargs), which beats defaults.
        return (env_settings, init_settings)

    # Derived paths -----------------------------------------------------------------
    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else self.paths.root / path

    def raw_dir(self) -> Path:
        return self.resolve(self.paths.raw_dir)

    def processed_corpus_dir(self) -> Path:
        return self.resolve(self.paths.processed_dir) / self.corpus.version

    def index_corpus_dir(self) -> Path:
        return self.resolve(self.paths.index_dir) / self.corpus.version

    def qdrant_path(self) -> Path:
        return self.resolve(self.qdrant.path) if self.qdrant.path else self.index_corpus_dir() / "qdrant"

    def hash(self) -> str:
        """Stable digest of the non-secret configuration, recorded in corpus manifests."""
        payload = self.model_dump(mode="json", exclude={"qdrant": {"api_key"}, "llm": {"api_key"}})
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

"""Model connectors. Provider names appear here and nowhere else."""

import os
import ipaddress
from urllib.parse import urlsplit

DEFAULT_LLM_PROVIDER = "ollama"
DEFAULT_EMBEDDING_PROVIDER = "ollama"
# Same embedder as legal_rag's Qdrant corpora, so Ollama keeps one embedding model resident.
DEFAULT_OLLAMA_EMBED_MODEL = "bge-m3"

# Keep embedding spaces in separate collections, even when dimensions match.
COLLECTION_NAMES = {
    ("ollama", "nomic-embed-text"): "docs_ollama_nomic",
}

# "none" disables a connector explicitly and is not an error. Ollama needs no
# credential, so without it there would be no way to run the app with AI turned
# off — and the graceful-degradation paths that behaviour feeds still exist.
DISABLED = "none"


def _llm_provider():
    return os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()


def _embedding_provider():
    return os.getenv("EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER).strip().lower()


DEFAULT_NUM_CTX = 12288  # configs/local.yaml llm.num_ctx


def _model_url():
    url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password or not parsed.hostname:
        raise RuntimeError("invalid_model_url")
    local = parsed.hostname == "localhost"
    if not local:
        try:
            local = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            pass
    if not local:
        allowed = os.getenv("ALLOW_USER_DOCUMENT_EGRESS", "").lower() in {"1", "true", "yes", "on"}
        if not allowed or parsed.scheme != "https":
            raise RuntimeError("document_egress_disabled")
    return url


def get_chat_model():
    """Return a chat model, or None when the provider is disabled."""
    provider = _llm_provider()

    if provider == DISABLED:
        return None

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        # Keep provider reasoning defaults unless explicitly overridden.
        # Response parsing supports both structured output and JSON text.
        options = {
            "model": os.getenv("OLLAMA_MODEL", "qwen3:4b"),
            "base_url": _model_url(),
            "client_kwargs": {"timeout": 120.0},
            "num_predict": 2048,
            # Sampling settings can be overridden for the configured model.
            "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.7")),
            "top_p": 0.8,
            "top_k": 20,
            # Ollama's own default is 4096 tokens and it drops the start of a longer prompt
            # silently (2026-09-27). The value must equal legal_rag's llm.num_ctx, or Ollama
            # reloads the model whenever the other client calls; so LEGAL_RAG__LLM__NUM_CTX
            # sets both, and OLLAMA_NUM_CTX overrides it for this app alone.
            "num_ctx": int(os.getenv("OLLAMA_NUM_CTX") or os.getenv("LEGAL_RAG__LLM__NUM_CTX") or DEFAULT_NUM_CTX),
        }
        reasoning = os.getenv("OLLAMA_REASONING")
        if reasoning is not None:
            options["reasoning"] = reasoning.strip().lower() in ("1", "true", "yes")
        return ChatOllama(**options)

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


def get_chat_descriptor():
    """Return provider and model identifiers without constructing a client."""
    provider = _llm_provider()
    if provider == "ollama":
        return provider, os.getenv("OLLAMA_MODEL", "qwen3:4b")
    if provider == DISABLED:
        return provider, ""
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


def get_embedding_model():
    """Return an embedding model, or None when the provider is disabled."""
    provider = _embedding_provider()

    if provider == DISABLED:
        return None

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_OLLAMA_EMBED_MODEL),
            base_url=_model_url(),
            client_kwargs={"timeout": 60.0},
            # Same window as legal_rag's embedding.query_num_ctx: Ollama reloads bge-m3
            # when clients disagree. Upload chunks are 1,000 characters, well under it.
            num_ctx=int(os.getenv("OLLAMA_EMBED_NUM_CTX", "1024")),
        )

    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider!r}")


def get_embedding_descriptor():
    provider = _embedding_provider()
    if provider == "ollama":
        return provider, os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_OLLAMA_EMBED_MODEL)
    if provider == DISABLED:
        return provider, ""
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider!r}")


def _normalise_model_id(model):
    """Normalize the default Ollama tag for collection lookup only."""
    model = model.strip()
    if model.endswith(":latest"):
        model = model[: -len(":latest")]
    return model


def get_collection_name():
    """Chroma collection dedicated to the active embedding provider and model."""
    provider, model = get_embedding_descriptor()
    if provider == DISABLED:
        return ""
    model = _normalise_model_id(model)
    for (table_provider, table_model), name in COLLECTION_NAMES.items():
        if table_provider == provider and _normalise_model_id(table_model) == model:
            return name
    safe_model = "".join(c if c.isalnum() else "_" for c in model).strip("_")
    return f"docs_{provider}_{safe_model}"


def describe_configuration():
    """Reporting helper for /health. Must not construct clients."""
    return {
        "llm_provider": _llm_provider(),
        "embedding_provider": _embedding_provider(),
        "collection": get_collection_name(),
    }

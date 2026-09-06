"""Model connectors. Provider names appear here and nowhere else."""

import os

DEFAULT_LLM_PROVIDER = "ollama"
DEFAULT_EMBEDDING_PROVIDER = "google"

# Chroma is dimension-locked per collection. Two providers at the same dimension
# (text-embedding-004 and nomic-embed-text are both 768) will be silently accepted
# into one collection and then compared across unrelated embedding spaces, which
# produces wrong neighbours with no error. The collection name therefore encodes
# the provider and model, so switching connectors creates a new collection.
COLLECTION_NAMES = {
    ("google", "models/text-embedding-004"): "docs_google_te004",
    ("ollama", "nomic-embed-text"): "docs_ollama_nomic",
}


def _llm_provider():
    return os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()


def _embedding_provider():
    return os.getenv("EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER).strip().lower()


def get_chat_model():
    """Return a chat model, or None when the provider is not configured."""
    provider = _llm_provider()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        # The `reasoning` kwarg is deliberately left unset. All three settings were
        # measured on qwen3:4b, and each one trades away something:
        #
        #   reasoning=False  thinking prose leaks into `content` with the <think>
        #                    tags stripped, so nothing downstream can recognise it.
        #   reasoning=True   `content` stays clean and thinking is isolated in
        #                    additional_kwargs["reasoning_content"] — but it
        #                    suppressed the structured-output tool call in 3 of 3
        #                    create_agent runs, so `structured_response` was never
        #                    populated.
        #   unset            `content` stays clean, thinking is dropped, and
        #                    `structured_response` populated in 2 of 3 runs.
        #
        # Unset is the only setting where the structured-output path works at all,
        # so it wins. That 2-of-3 is also why extract_agent_sections() in
        # langchain_config.py must keep its JSON fallback: a 4B does not honour a
        # response schema reliably enough to depend on. Set OLLAMA_REASONING=true
        # to trade structured output for thinking visibility in traces.
        options = {
            "model": os.getenv("OLLAMA_MODEL", "qwen3:4b"),
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            # Qwen3 degrades and repeats under greedy decoding. Do not use 0.
            "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.7")),
            "top_p": 0.8,
            "top_k": 20,
        }
        reasoning = os.getenv("OLLAMA_REASONING")
        if reasoning is not None:
            options["reasoning"] = reasoning.strip().lower() in ("1", "true", "yes")
        return ChatOllama(**options)

    if provider == "google":
        if not os.getenv("GOOGLE_API_KEY"):
            return None
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.0-flash"),
            temperature=0,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


def get_embedding_model():
    """Return an embedding model, or None when the provider is not configured."""
    provider = _embedding_provider()

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    if provider == "google":
        if not os.getenv("GOOGLE_API_KEY"):
            return None
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=os.getenv("GOOGLE_EMBED_MODEL", "models/text-embedding-004")
        )

    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider!r}")


def get_embedding_descriptor():
    provider = _embedding_provider()
    if provider == "ollama":
        return provider, os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    if provider == "google":
        return provider, os.getenv("GOOGLE_EMBED_MODEL", "models/text-embedding-004")
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider!r}")


def _normalise_model_id(model):
    """Collapse the spellings of one model onto a single key.

    Google accepts both `models/text-embedding-004` and `text-embedding-004`, and
    Ollama accepts both `nomic-embed-text` and `nomic-embed-text:latest`. They are
    the same embedding space each time, so they must not produce two collections:
    that would silently re-embed and hide everything ingested under the other
    spelling. Lookup path only. get_embedding_descriptor() keeps reporting the raw
    spelling, because that is what is handed to the SDK.
    """
    model = model.strip()
    if model.startswith("models/"):
        model = model[len("models/"):]
    if model.endswith(":latest"):
        model = model[: -len(":latest")]
    return model


def get_collection_name():
    """Chroma collection dedicated to the active embedding provider and model."""
    provider, model = get_embedding_descriptor()
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

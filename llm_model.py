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

        # reasoning=True keeps `content` clean and isolates Qwen3's thinking in
        # additional_kwargs["reasoning_content"]. reasoning=False is NOT the safe
        # choice: it strips the <think> tags and leaves the reasoning prose inside
        # content, where nothing downstream can recognise it. Verified on qwen3:4b.
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen3:4b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            reasoning=True,
            # Qwen3 degrades and repeats under greedy decoding. Do not use 0.
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.7")),
            top_p=0.8,
            top_k=20,
        )

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


def get_collection_name():
    """Chroma collection dedicated to the active embedding provider and model."""
    descriptor = get_embedding_descriptor()
    if descriptor in COLLECTION_NAMES:
        return COLLECTION_NAMES[descriptor]
    provider, model = descriptor
    safe_model = "".join(c if c.isalnum() else "_" for c in model).strip("_")
    return f"docs_{provider}_{safe_model}"


def describe_configuration():
    """Reporting helper for /health. Must not construct clients."""
    return {
        "llm_provider": _llm_provider(),
        "embedding_provider": _embedding_provider(),
        "collection": get_collection_name(),
    }

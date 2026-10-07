import os
import uuid
from threading import Lock

_ingest_lock = Lock()
from datetime import datetime, timezone

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm_model import get_collection_name, get_embedding_model

# Define directories. Resolved at call time, never at import: importing this
# module must not touch disk or construct a client.
PERSIST_DIRECTORY = "./chroma_db"


def get_persist_directory():
    """Where Chroma lives. RAG_PERSIST_DIR points tests at a temporary directory."""
    return os.getenv("RAG_PERSIST_DIR", PERSIST_DIRECTORY)

# --- LangSmith tracing ---------------------------------------------------
#
# Off unless LANGSMITH_TRACING is explicitly truthy. Tracing ships the full
# prompt, the retrieved document text and the model's answer to LangSmith, and
# this app holds users' legal documents -- so it is opt-in, per deployment,
# and never the default.

TRUTHY = ("1", "true", "yes", "on")


def tracing_enabled():
    """True only when tracing was deliberately switched on.

    Empty string counts as off: .env.example ships `LANGSMITH_TRACING=` and
    load_dotenv() puts that empty value into the environment, so a plain
    presence check would enable tracing for anyone who copied the file.
    """
    return os.getenv("LANGSMITH_TRACING", "").strip().lower() in TRUTHY


def tracing_status():
    """What /health and startup logging can say without leaking the API key."""
    return {
        "enabled": tracing_enabled(),
        "project": os.getenv("LANGSMITH_PROJECT", "") or "default",
        "api_key_present": bool(os.getenv("LANGSMITH_API_KEY", "").strip()),
    }


def get_vector_store():
    """Build Chroma on demand with a collection scoped to the embedding model."""
    embedding_model = get_embedding_model()
    if embedding_model is None:
        return None
    return Chroma(
        collection_name=get_collection_name(),
        persist_directory=get_persist_directory(),
        embedding_function=embedding_model,
    )


def ingest_text_to_vector_store(text, source_name, metadata=None, user_id=None):
    """Add extracted document text to the persistent RAG store for later chat follow-up."""
    if not user_id:
        return {"ok": False, "message": "user_id is required.", "chunk_count": 0}
    vector_store = get_vector_store()
    if vector_store is None:
        return {"ok": False, "message": "Embedding model is not configured.", "chunk_count": 0}

    if not isinstance(text, str) or len(text) > 20000 or not text.strip():
        return {"ok": False, "message": "No text was available to ingest.", "chunk_count": 0}

    document_metadata = {
        "source": source_name or "uploaded_document",
    }
    if metadata:
        document_metadata.update(metadata)
    document_id = uuid.uuid4().hex
    ingested_at = datetime.now(timezone.utc).isoformat()
    document_metadata.update({
        "document_id": document_id,
        "ingested_at": ingested_at,
        "user_id": user_id,
    })

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents([
        Document(page_content=text, metadata=document_metadata)
    ])

    if not chunks:
        return {"ok": False, "message": "Document text could not be split into chunks.", "chunk_count": 0}

    # Use one web worker until a shared storage admission lock is implemented.
    # Refuse new data when full; never evict existing documents implicitly.
    with _ingest_lock:
        existing = vector_store.get(where={"user_id": user_id}, include=[])
        if len(existing.get("ids", [])) + len(chunks) > 5000:
            return {"ok": False, "message": "workspace_storage_limit", "chunk_count": 0}
        vector_store.add_documents(chunks)

    if hasattr(vector_store, "persist"):
        vector_store.persist()

    return {
        "ok": True,
        "message": f"Saved {len(chunks)} chunks to the knowledge base.",
        "chunk_count": len(chunks),
        "document_id": document_id,
        "ingested_at": ingested_at,
    }

def get_knowledge_base_status(user_id):
    """Return a compact summary of documents currently stored in Chroma."""
    vector_store = get_vector_store()
    if vector_store is None:
        return {
            "available": False,
            "chunk_count": 0,
            "sources": [],
        }

    try:
        stored = vector_store.get(where={"user_id": user_id}, include=["metadatas"])
        metadatas = stored.get("metadatas", []) or []
        sources_by_name = {}
        for metadata in metadatas:
            metadata = metadata or {}
            source = metadata.get("source") or "Unknown source"
            if source not in sources_by_name:
                sources_by_name[source] = {
                    "source": source,
                    "chunk_count": 0,
                    "uploaded_via": metadata.get("uploaded_via", ""),
                    "analysis_mode": metadata.get("analysis_mode", ""),
                }
            sources_by_name[source]["chunk_count"] += 1

        sources = sorted(sources_by_name.values(), key=lambda item: item["source"].lower())
        return {
            "available": bool(metadatas),
            "chunk_count": len(metadatas),
            "sources": sources,
        }
    except Exception as e:
        return {
            "available": False,
            "chunk_count": 0,
            "sources": [],
            "error": "knowledge_base_failed",
        }

def clear_knowledge_base(user_id):
    """Remove all chunks from the active Chroma collection without deleting files manually."""
    vector_store = get_vector_store()
    if vector_store is None:
        return {"ok": True, "message": "Knowledge base is already empty.", "deleted_count": 0}

    try:
        stored = vector_store.get(where={"user_id": user_id})
        ids = stored.get("ids", []) or []
        if ids:
            vector_store.delete(ids=ids)
            if hasattr(vector_store, "persist"):
                vector_store.persist()
        return {"ok": True, "message": f"Removed {len(ids)} chunks from the knowledge base.", "deleted_count": len(ids)}
    except Exception as e:
        return {"ok": False, "message": "Failed to clear knowledge base.", "deleted_count": 0}

def search_knowledge_base(user_id, query, k=6):
    """Similarity search over the user's chunks. Returns a list of {source, score, quote, chunk_index}."""
    vector_store = get_vector_store()
    if vector_store is None or not user_id or not query or not query.strip():
        return {"ok": vector_store is not None, "results": []}
    try:
        pairs = vector_store.similarity_search_with_relevance_scores(
            query, k=k, filter={"user_id": user_id}
        )
        results = []
        for doc, score in pairs:
            metadata = doc.metadata or {}
            results.append({
                "source": metadata.get("source") or "Unknown source",
                "score": max(0.0, min(1.0, float(score))),
                "quote": (doc.page_content or "").strip(),
                "analysis_mode": metadata.get("analysis_mode", ""),
            })
        return {"ok": True, "results": results}
    except Exception as e:
        return {"ok": False, "results": [], "error": "knowledge_base_failed"}

def delete_knowledge_base_source(user_id, source):
    """Remove every chunk that belongs to one uploaded file for this user."""
    vector_store = get_vector_store()
    if vector_store is None:
        return {"ok": True, "deleted_count": 0}
    if not user_id or not source:
        return {"ok": False, "message": "user_id and source are required.", "deleted_count": 0}
    try:
        stored = vector_store.get(where={"$and": [{"user_id": user_id}, {"source": source}]})
        ids = stored.get("ids", []) or []
        if ids:
            vector_store.delete(ids=ids)
            if hasattr(vector_store, "persist"):
                vector_store.persist()
        return {"ok": True, "deleted_count": len(ids)}
    except Exception as e:
        return {"ok": False, "message": "Failed to delete source.", "deleted_count": 0}



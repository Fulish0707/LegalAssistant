import os
import uuid
from datetime import datetime, timezone

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
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


def get_documents_dir():
    return os.getenv("RAG_DOCUMENTS_DIR", "./documents")


def load_rag_documents(source_folder):
    supported_loaders = {
        ".pdf": lambda path: PyPDFLoader(path).load(),
        ".txt": lambda path: TextLoader(path, encoding="utf-8").load(),
    }
    loaded_documents = []
    for root, _, files in os.walk(source_folder):
        for filename in files:
            extension = os.path.splitext(filename)[1].lower()
            loader_factory = supported_loaders.get(extension)
            if not loader_factory:
                continue
            file_path = os.path.join(root, filename)
            try:
                loaded_documents.extend(loader_factory(file_path))
            except Exception as e:
                print(f"Could not load RAG document {file_path}: {e}")
    return loaded_documents

def get_document_source_folder():
    doc_folder = get_documents_dir()
    if os.path.exists(doc_folder) and any(os.scandir(doc_folder)):
        return doc_folder
    legacy_doc_folder = "./doc_folder"
    if os.path.exists(legacy_doc_folder) and any(os.scandir(legacy_doc_folder)):
        print(f"No files found in {doc_folder}. Using legacy folder {legacy_doc_folder} for RAG ingestion.")
        return legacy_doc_folder
    return doc_folder

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


def ingest_source_folder(user_id=None):
    """Ingest ./documents into the provider-scoped collection.

    Explicit by design: this used to run at import, which made the collection a
    function of import order. Callers invoke it when they actually want ingestion.

    user_id is what makes the result reachable. Every read path -- search,
    status, the per-account retriever -- filters on user_id, so chunks ingested
    without one are write-only: stored in Chroma and invisible to the app. The
    old import-time block never stamped it either.

    Not idempotent. The old block ran only when chroma_db/ was absent; calling
    this twice ingests the same files twice.
    """
    vector_store = get_vector_store()
    if vector_store is None:
        return {"ok": False, "message": "Embedding model is not configured.", "chunk_count": 0}

    doc_folder = get_documents_dir()
    if not os.path.exists(doc_folder):
        os.makedirs(doc_folder)
        print(f"Created documents directory at {doc_folder}")

    source_folder = get_document_source_folder()
    documents = load_rag_documents(source_folder)
    print(f"Loaded {len(documents)} documents from {source_folder}")
    if not documents:
        return {"ok": False, "message": f"No supported RAG documents found in {source_folder}.", "chunk_count": 0}

    if user_id:
        for document in documents:
            document.metadata = {**(document.metadata or {}), "user_id": user_id}

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    if not chunks:
        return {"ok": False, "message": "Documents could not be split into chunks.", "chunk_count": 0}

    vector_store.add_documents(chunks)
    if hasattr(vector_store, "persist"):
        vector_store.persist()
    print(f"Ingested {len(chunks)} chunks into {get_collection_name()}")
    return {"ok": True, "message": f"Ingested {len(chunks)} chunks.", "chunk_count": len(chunks)}


def ingest_text_to_vector_store(text, source_name, metadata=None, user_id=None):
    """Add extracted document text to the persistent RAG store for later chat follow-up."""
    if not user_id:
        return {"ok": False, "message": "user_id is required.", "chunk_count": 0}
    vector_store = get_vector_store()
    if vector_store is None:
        return {"ok": False, "message": "Embedding model is not configured.", "chunk_count": 0}

    if not text or not text.strip():
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
            "error": str(e),
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
        return {"ok": False, "message": f"Failed to clear knowledge base: {e}", "deleted_count": 0}

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
        return {"ok": False, "results": [], "error": str(e)}

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
        return {"ok": False, "message": f"Failed to delete source: {e}", "deleted_count": 0}

def get_user_retriever(user_id):
    """Build a retriever for one account without mutating process-global state."""
    vector_store = get_vector_store()
    if vector_store is None or not user_id:
        return None
    return vector_store.as_retriever(search_kwargs={"k": 3, "filter": {"user_id": user_id}})


def retrieve_for_user(user_id, query, k=4):
    """Similarity search scoped to one account. Seam for tests."""
    store = get_vector_store()
    if store is None or not user_id:
        return []
    return store.similarity_search(query, k=k, filter={"user_id": user_id})



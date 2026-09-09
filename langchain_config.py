from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from llm_model import (
    get_chat_descriptor,
    get_chat_model,
    get_collection_name,
    get_embedding_descriptor,
    get_embedding_model,
)
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from langchain_core.tools import tool
from llm_prompt import KOREAN_LEGAL_POLICY, CHAT_OUTPUT_INSTRUCTIONS
from pydantic import BaseModel, Field

# Define directories. Resolved at call time, never at import: importing this
# module must not touch disk or construct a client.
PERSIST_DIRECTORY = "./chroma_db"


def get_persist_directory():
    """Where Chroma lives. Env-overridable so the eval harness can be sandboxed.

    evals/run_eval.py must not write its fixture vectors into the real user
    knowledge base, and deleting them afterwards by id is a weaker guarantee
    than never sharing the store. RAG_PERSIST_DIR points it at a temporary
    directory instead; unset, nothing changes.
    """
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


def run_metadata(user_id=None):
    """Attach provider, model, collection, and account identifiers to each run."""
    llm_provider, llm_model = get_chat_descriptor()
    embedding_provider, embedding_model = get_embedding_descriptor()
    return {
        "thread_id": user_id or "guest",
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "collection": get_collection_name(),
    }


def run_tags(user_id=None):
    """Tags are what a LangSmith comparison view filters and groups on."""
    llm_provider, llm_model = get_chat_descriptor()
    return [f"provider:{llm_provider}", f"model:{llm_model}"]


def build_run_config(user_id=None, extra=None):
    """The runnable config for one agent run: thread, metadata and tags.

    Built in one place so agent.stream() and any future agent.invoke() cannot
    drift apart on what a run is tagged with.
    """
    config = {"metadata": run_metadata(user_id), "tags": run_tags(user_id)}
    if user_id:
        config["configurable"] = {"thread_id": user_id}
    if extra:
        config.update(extra)
    return config



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


def build_rag_tool(user_id):
    """Build an account-scoped tool returning document excerpts and source names."""

    @tool("search_documents")
    def search_documents(query: str) -> str:
        """Search the signed-in user's uploaded documents. Returns excerpts with sources."""
        documents = retrieve_for_user(user_id, query)
        if not documents:
            return "No matching passages were found in the user's documents."

        seen = set()
        blocks = []
        for document in documents:
            body = (document.page_content or "").strip()
            if not body or body in seen:
                continue
            seen.add(body)
            source = (document.metadata or {}).get("source") or "Unknown source"
            blocks.append(f"[source: {source}]\n{body}")

        return "\n\n".join(blocks)

    return search_documents


AGENT_PREFIX = KOREAN_LEGAL_POLICY + CHAT_OUTPUT_INSTRUCTIONS

@dataclass
class AgentContext:
    """Typed request context. Replaces closure-captured globals."""

    user_id: Optional[str] = None


class AgentSections(BaseModel):
    """Structured legal answer fields consumed by the frontend."""

    chat_response: str = Field(description="The nonempty Korean-law conversational answer with sources and material caveats.")
    statement_suggestion: str = Field(default="", description="Legal document drafts or revisions (서류 초안 제안), such as 내용증명 or agreements.")
    job_statement_suggestion: str = Field(default="", description="Legal application or filing drafts (신청서 제안).")
    report_suggestion: str = Field(default="", description="Legal document review, clause risks, and proposed revisions (검토 의견 제안).")


SECTION_FIELDS = (
    "chat_response",
    "statement_suggestion",
    "job_statement_suggestion",
    "report_suggestion",
)


def extract_agent_sections(result):
    """Read schema output or JSON message content; return None for marker-parser fallback."""
    if not isinstance(result, dict):
        return None

    structured = result.get("structured_response")
    if structured is not None:
        if isinstance(structured, BaseModel):
            structured = structured.model_dump()
        if isinstance(structured, dict) and structured.get("chat_response"):
            return {field: str(structured.get(field) or "").strip() for field in SECTION_FIELDS}

    messages = result.get("messages") or []
    content = ""
    for message in reversed(messages):
        text = getattr(message, "content", None)
        if isinstance(text, str) and text.strip():
            content = text.strip()
            break
    if not content:
        return None

    # Parse the raw content before any cleaning. clean_agent_output() unwraps code
    # fences and rewrites text, which can break otherwise valid JSON.
    fenced = re.match(r"^```(?:json)?\s*\n(.*?)\n?```$", content, re.DOTALL)
    if fenced:
        content = fenced.group(1).strip()
    if not content.startswith("{"):
        return None
    try:
        payload = json.loads(content)
    except ValueError:
        return None
    if not isinstance(payload, dict) or not payload.get("chat_response"):
        return None
    return {field: str(payload.get(field) or "").strip() for field in SECTION_FIELDS}


def build_agent(user_id=None, checkpointer=None):
    """Create a request-scoped agent and only expose RAG to signed-in accounts."""
    # Imported here, not at module scope: importing langchain_config must stay
    # cheap and inert, and langgraph is a heavy import.
    from langchain.agents import create_agent

    model = get_chat_model()
    if model is None:
        return None

    if not hasattr(model, "bind_tools"):
        # The agent requires a model that supports tool binding.
        raise RuntimeError(
            f"LLM_PROVIDER={os.getenv('LLM_PROVIDER')} returned a model without "
            "bind_tools; create_agent cannot run a tool loop with it."
        )

    # Document retrieval is available only to signed-in accounts.
    tools = [build_rag_tool(user_id)] if user_id else []

    return create_agent(
        model,
        tools=tools,
        system_prompt=AGENT_PREFIX,
        response_format=AgentSections,
        context_schema=AgentContext,
        checkpointer=checkpointer,
    )

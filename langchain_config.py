from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_chroma import Chroma
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_classic.agents import initialize_agent
from langchain_core.documents import Document
from llm_model import get_chat_model, get_collection_name, get_embedding_model
import os
import uuid
from datetime import datetime, timezone
from langchain_core.tools import tool

# Define directories. Resolved at call time, never at import: importing this
# module must not touch disk or construct a client.
PERSIST_DIRECTORY = "./chroma_db"


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
    """Build the Chroma handle on demand. Never at import.

    collection_name is the only thing keeping providers apart. text-embedding-004
    and nomic-embed-text are both 768-dimensional, so Chroma would accept vectors
    from either into one collection and then compute neighbours across two
    unrelated embedding spaces -- wrong results, no exception. A bare Chroma(...)
    would land in the legacy `langchain` collection, which is Google-embedded.
    """
    embedding_model = get_embedding_model()
    if embedding_model is None:
        return None
    return Chroma(
        collection_name=get_collection_name(),
        persist_directory=PERSIST_DIRECTORY,
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
    """A deterministic retrieval tool scoped to one account.

    Every step here is code, not model judgement. That matters more with a 4B
    model: what the model cannot decide, it cannot get wrong. The old version
    retrieved twice per call -- once directly, then again inside the QA chain --
    and spent an extra LLM call summarising what the agent was about to read
    anyway. This retrieves once and hands back the passages.
    """

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


AGENT_PREFIX = """You are PS_Agent, a practical AI work companion for students and professionals.

Your main jobs are:
- Help users draft and improve university personal statements, job application statements, and professional reports.
- Summarize uploaded or retrieved documents clearly.
- Answer general questions in a ChatGPT-like way.
- Use Search when the user asks for current, external, or broadly factual information.
- Use search_documents when the user asks about the local document database or source documents.

When the task is about writing, return useful sections when relevant:
[CHAT] for the conversational answer.
[STATEMENT_SUGGESTION] for university personal statement improvements.
[JOB_STATEMENT_SUGGESTION] for job application statement improvements.
[REPORT_SUGGESTION] for report structure, clarity, and business/audience improvements.

Follow the user's language. Prefer Korean when the user writes in Korean; otherwise use English."""

# Load SerpAPI tool only when configured so the app can still run locally without web search.
serpapi_api_key = os.getenv("SERPAPI_API_KEY")
serpapi_tools = []
if serpapi_api_key:
    try:
        serpapi_tools = load_tools(
            tool_names=["serpapi"],
            serpapi_api_key=serpapi_api_key
        )
    except Exception as e:
        print(f"Search tool could not be initialized: {e}")
else:
    print("SERPAPI_API_KEY not found. Web search is disabled until it is configured.")

def build_agent(user_id=None):
    """Create a request-scoped agent and only expose RAG to signed-in accounts."""
    llm = get_chat_model()
    if llm is None:
        return None

    tools = list(serpapi_tools)
    if user_id:
        tools.insert(0, build_rag_tool(user_id))
    return initialize_agent(
        tools=tools,
        llm = llm,
        agent="conversational-react-description",
        streaming = True,
        callbacks = [StreamingStdOutCallbackHandler()],
        verbose = True,
        handle_parsing_errors = True,
        temperature = 0.0,
        agent_kwargs={"prefix": AGENT_PREFIX}
    )

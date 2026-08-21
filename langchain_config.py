from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_chroma import Chroma
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_classic.agents import initialize_agent
from langchain_classic.chains import RetrievalQA
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.documents import Document
from llm_model import embedding_model, llm
import os
import uuid
from datetime import datetime, timezone
from langchain_core.tools import Tool
from dotenv import load_dotenv

load_dotenv() # Load variables from .env file

# Define directories
persis_directory = "./chroma_db"
doc_folder = os.getenv("RAG_DOCUMENTS_DIR", "./documents")

# Create documents directory if it doesn't exist
if not os.path.exists(doc_folder):
    os.makedirs(doc_folder)
    print(f"Created documents directory at {doc_folder}")

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
    if os.path.exists(doc_folder) and any(os.scandir(doc_folder)):
        return doc_folder
    legacy_doc_folder = "./doc_folder"
    if os.path.exists(legacy_doc_folder) and any(os.scandir(legacy_doc_folder)):
        print(f"No files found in {doc_folder}. Using legacy folder {legacy_doc_folder} for RAG ingestion.")
        return legacy_doc_folder
    return doc_folder

# Check if the vector store already exists
if embedding_model is None:
    print("Embedding model is unavailable. RAG vector store is disabled until GOOGLE_API_KEY is configured.")
    vector_store = None
elif os.path.exists(persis_directory):
    print(f"Loading existing vector store from {persis_directory}")
    vector_store = Chroma(
        persist_directory = persis_directory,
        embedding_function = embedding_model
    )
else:
    print(f"Creating new vector store in {persis_directory}")
    
    try:
        source_folder = get_document_source_folder()
        documents = load_rag_documents(source_folder)
        print(f"Loaded {len(documents)} documents from {source_folder}")

        if len(documents) == 0:
            print(f"Warning: No supported RAG documents found in {source_folder}")
            vector_store = None
        else:
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            print(f"Split into {len(chunks)} chunks")

            # Create and persist vector store
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory=persis_directory
            )
            if hasattr(vector_store, "persist"):
                vector_store.persist()
            print(f"Vector store created and persisted in {persis_directory}")

    except Exception as e:
        print(f"An error occurred during vector store creation: {e}")
        vector_store = None

def ingest_text_to_vector_store(text, source_name, metadata=None, user_id=None):
    """Add extracted document text to the persistent RAG store for later chat follow-up."""
    global vector_store

    if not user_id:
        return {"ok": False, "message": "user_id is required.", "chunk_count": 0}
    if embedding_model is None:
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

    if vector_store is None:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persis_directory
        )
    else:
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

def get_user_retriever(user_id):
    """Build a retriever for one account without mutating process-global state."""
    if vector_store is None or not user_id:
        return None
    return vector_store.as_retriever(search_kwargs={"k": 3, "filter": {"user_id": user_id}})


def build_rag_tool(user_id):
    """Create a RAG tool whose retrieval is permanently scoped to one account."""
    def rag_tool_func(query):
        retriever = get_user_retriever(user_id)
        if not retriever:
            return "RAG가 현재 사용 불가능합니다."
        try:
            # 1. 기본 검색
            docs = retriever.invoke(query)
            basic_results = "\n".join([doc.page_content for doc in docs])
            
            # 2. RetrievalQA 체인 생성
            qa_chain = load_qa_chain(
                llm = llm,
                chain_type="stuff"
            )

            qa = RetrievalQA(
                combine_documents_chain=qa_chain,
                retriever=retriever,
                return_source_documents=True
            )
            
            # 3. 질의응답 실행
            qa_result = qa({"query": query})
            
            # 4. 결과 결합
            combined_response = f"""
관련 문서:
{basic_results}

질문에 대한 답변:
{qa_result['result']}

참고한 문서 수: {len(qa_result['source_documents'])}
"""
            return combined_response
        except Exception as e:
            print(f"RAG 처리 중 오류 발생: {e}")
            return "RAG 처리 중 오류가 발생했습니다."

    return Tool(
        name="RAG Retriever",
        func=rag_tool_func,
        description="Use this tool to search the signed-in user's internal document database and get detailed answers to questions.",
    )

AGENT_PREFIX = """You are PS_Agent, a practical AI work companion for students and professionals.

Your main jobs are:
- Help users draft and improve university personal statements, job application statements, and professional reports.
- Summarize uploaded or retrieved documents clearly.
- Answer general questions in a ChatGPT-like way.
- Use Search when the user asks for current, external, or broadly factual information.
- Use RAG Retriever when the user asks about the local document database or source documents.

When the task is about writing, return useful sections when relevant:
[CHAT] for the conversational answer.
[STATEMENT_SUGGESTION] for university personal statement improvements.
[JOB_STATEMENT_SUGGESTION] for job application statement improvements.
[REPORT_SUGGESTION] for report structure, clarity, and business/audience improvements.

Follow the user's language. Prefer Korean when the user writes in Korean; otherwise use English."""

# Load SerpAPI tool only when configured so the app can still run locally without web search.
serpapi_api_key = os.getenv("SERPAPI_API_KEY")
serpapi_tools = []
if serpapi_api_key and llm is not None:
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


if llm is None:
    print("LLM is unavailable. Chat agent is disabled until GOOGLE_API_KEY is configured.")

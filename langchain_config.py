from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain.chains import RetrievalQA, StuffDocumentsChain
from langchain.chains.question_answering import load_qa_chain
from llm_model import embedding_model, llm
from llm_prompt import prompt
import os
from langchain.tools import Tool
from dotenv import load_dotenv

load_dotenv() # Load variables from .env file

# Define directories
persis_directory = "./chroma_db"
doc_folder = "./documents"  # 문서 폴더 경로

# Create documents directory if it doesn't exist
if not os.path.exists(doc_folder):
    os.makedirs(doc_folder)
    print(f"Created documents directory at {doc_folder}")

# Check if the vector store already exists
if os.path.exists(persis_directory):
    print(f"Loading existing vector store from {persis_directory}")
    vector_store = Chroma(
        persist_directory = persis_directory,
        embedding_function = embedding_model
    )
else:
    print(f"Creating new vector store in {persis_directory}")
    
    try:
        # Load all documents from the directory
        loader = DirectoryLoader(
            doc_folder,
            glob="**/*.*",  # 모든 파일 형식 로드
            loader_cls={
                ".pdf": PyPDFLoader,
                ".txt": TextLoader,
            }
        )
        documents = loader.load()
        print(f"Loaded {len(documents)} documents from {doc_folder}")

        if len(documents) == 0:
            print(f"Warning: No documents found in {doc_folder}")
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
            vector_store.persist()
            print(f"Vector store created and persisted in {persis_directory}")

    except Exception as e:
        print(f"An error occurred during vector store creation: {e}")
        vector_store = None

# Create a retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3}) if vector_store else None

# Create a memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define a RAG tool using the retriever
def rag_tool_func(query):
    if retriever:
        try:
            # 1. 기본 검색
            docs = retriever.invoke(query)
            basic_results = "\n".join([doc.page_content for doc in docs])
            
            # 2. RetrievalQA 체인 생성
            qa_chain = load_qa_chain(
                llm=llm,
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
    else:
        return "RAG가 현재 사용 불가능합니다."

rag_tool = Tool(
    name="RAG Retriever",
    func=rag_tool_func,
    description="Use this tool to search the internal document database and get detailed answers to questions."
)

# Load SerpAPI tool
serpapi_tools = load_tools(
    tool_names=["serpapi"],
    serpapi_api_key=os.getenv("SERPAPI_API_KEY")
)

# Combine tools
tools = [rag_tool] + serpapi_tools

# Create agent with custom prompt
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="conversational-react-description",
    verbose=True,
    memory=memory,
    system_message=prompt.messages[0].prompt.template
)
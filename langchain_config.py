from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from llm_model import embedding_model, llm
from llm_prompt import prompt
import os
from langchain.tools import Tool
from dotenv import load_dotenv

load_dotenv() # Load variables from .env file

# Define a directory to save the vector store
persis_directory = "./chroma_db"

# Check if the vector store already exists
if os.path.exists(persis_directory):
    print(f"Loading existing vector store from {persis_directory}")
    vector_store = Chroma(
        persist_directory = persis_directory,
        embedding_function = embedding_model
    )
else:
    print(f"Creating new vector store in {persis_directory}")
    file_path = "personal_statement_sample.pdf"

    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents from {file_path}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")

        # Create the vector store and persist it
        vector_store = Chroma.from_documents(
            documents = chunks,
            embedding = embedding_model,
            persist_directory = persis_directory
        )
        vector_store.persist()
        print(f"Vector store created and persisted in {persis_directory}")

    except FileNotFoundError:
        print(f"Error: Document file not found at {file_path}. RAG will not be available.")
        vector_store = None
    except Exception as e:
        print(f"An error occurred during vector store creation: {e}")
        vector_store = None

# Create a retriever
retriever = vector_store.as_retriever(search_kwargs = {"k" : 3}) if vector_store else None

# Create a memory
memory = ConversationBufferMemory(memory_key = "chat_history", return_messages = True)

# Define a RAG tool using the retriever

def rag_tool_func(query):
    if retriever:
        docs = retriever.invoke(query)
        return "\n".join([doc.page_content for doc in docs])
    else:
        return "RAG is not available."

rag_tool = Tool(
    name="RAG Retriever",
    func=rag_tool_func,
    description="Use this tool to search the internal document database for relevant information."
)

# Load SerpAPI tool
serpapi_tools = load_tools(
    tool_names=["serpapi"],
    serpapi_api_key=os.getenv("SERPAPI_API_KEY")
)

# Combine tools
tools = [rag_tool] + serpapi_tools

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    system_message=prompt.messages[0].prompt.template
)
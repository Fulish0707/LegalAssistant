import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Make sure API key is set
load_dotenv() # if using .env file
google_api_key = os.getenv("GOOGLE_API_KEY")

if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key

    llm = ChatGoogleGenerativeAI(
        temperature = 0,
        model = "gemini-2.0-flash"
        )
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
        )
else:
    print("GOOGLE_API_KEY is not configured. LLM and embedding features are disabled until .env is updated.")
    llm = None
    embedding_model = None

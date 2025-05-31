import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Make sure API key is set
load_dotenv() # if using .env file
google_api_key = os.getenv("GOOGLE_API_KEY")

os.environ["GOOGLE_API_KEY"] = google_api_key

llm = ChatGoogleGenerativeAI(
    temperature = 0,
    model = "gemini-2.0-flash"
    )
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
    )
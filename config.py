import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is missing!\n"
        "   → Create a .env file in the project root with:\n"
        "       GOOGLE_API_KEY=your_key_here\n"
        "   → Get a free key at https://aistudio.google.com/apikey"
    )

CHROMA_PERSIST_DIR: str = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME: str = "rag_chat_collection"

EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Free, local sentence-transformers model
LLM_MODEL: str = "gemini-2.5-flash"
LLM_TEMPERATURE: float = 0.3
RETRIEVER_K: int = 5  # Number of chunks to retrieve

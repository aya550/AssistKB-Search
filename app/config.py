import os
from dotenv import load_dotenv

load_dotenv()

VECTOR_STORE = os.getenv("VECTOR_STORE", "qdrant")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "assistkb")

EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))

CHUNKS_PATH = os.getenv("CHUNKS_PATH", "corpus/chunks.jsonl")
TOP_K = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
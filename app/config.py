"""Configuration centralisee, lue depuis l'environnement (.env).

Tous les etages du pipeline importent leurs reglages ici. On evite ainsi
les valeurs magiques dispersees dans le code.
"""
import os

from dotenv import load_dotenv

load_dotenv()  # charge .env s'il existe (ignore en silence sinon)


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


# --- Embeddings (local) ----------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = _int("EMBED_DIM", 384)

# --- Vector store ----------------------------------------------------------
VECTOR_STORE = os.getenv("VECTOR_STORE", "qdrant")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "assistkb")

# --- Corpus / chunking -----------------------------------------------------
CORPUS_DIRS = [
    d.strip()
    for d in os.getenv("CORPUS_DIRS", "socle-commun/corpus/seed,corpus/raw").split(",")
    if d.strip()
]
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "corpus/chunks.jsonl")
CHUNK_SIZE = _int("CHUNK_SIZE", 800)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 120)

# --- Retrieval -------------------------------------------------------------
TOP_K = _int("TOP_K", 5)
# Seuil regle empiriquement (cf. compte rendu, section R3) : 0.45 = meilleur
# compromis pertinence/refus sur le corpus seed (bloque le hors-sujet sans
# trop sacrifier les questions limites). Surchargeable via .env.
SIMILARITY_THRESHOLD = _float("SIMILARITY_THRESHOLD", 0.45)
REFUSAL_MESSAGE = os.getenv(
    "REFUSAL_MESSAGE",
    "Je ne dispose pas d'information suffisante dans le corpus pour repondre "
    "a cette question de maniere fiable.",
)

# --- LLM -------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

"""
Central configuration. Every other module imports from here instead of
hardcoding values. Change chunk size, model names, etc. in exactly one place.
"""
import os
from dotenv import load_dotenv

# Loads variables from the .env file into the environment
load_dotenv()

_openai_api_key = os.getenv("OPENAI_API_KEY")
if not _openai_api_key:
    raise ValueError(
        "OPENAI_API_KEY not found. Add it to your .env file at the project root."
    )
OPENAI_API_KEY: str = _openai_api_key

# --- Models ---
EMBEDDING_MODEL = "text-embedding-3-small"  # cheap, fast, good enough for most RAG
CHAT_MODEL = "gpt-4o-mini"                  # cheap, fast chat model; swap for gpt-4o if needed

# --- Chunking ---
CHUNK_SIZE = 1000        # characters per chunk (roughly ~200-250 tokens)
CHUNK_OVERLAP = 150      # overlap between chunks so we don't cut a sentence mid-thought

# --- Retrieval ---
TOP_K = 4                # how many chunks to retrieve per query

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")

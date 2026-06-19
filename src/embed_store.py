"""
Embedding + Vector Store module.

Responsible for: converting text chunks into embeddings (vectors), building
a FAISS index from them, and saving/loading it from disk.
"""
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import OPENAI_API_KEY, EMBEDDING_MODEL, VECTORSTORE_DIR
from src.ingest import run_ingestion

INDEX_NAME = "faiss_index"


def get_embedding_model():
    """
    Returns the OpenAI embeddings client — converts text into a vector of
    ~1536 numbers representing its meaning. Similar meanings land close
    together in this vector space, which is what makes semantic search work.
    """
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )


def build_vectorstore(chunks):
    """
    Embeds every chunk and builds a FAISS index — a data structure optimized
    for fast nearest-neighbor search over vectors.
    """
    embeddings = get_embedding_model()
    print(f"Embedding {len(chunks)} chunks using {EMBEDDING_MODEL}...")

    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    save_path = os.path.join(VECTORSTORE_DIR, INDEX_NAME)
    vectorstore.save_local(save_path)
    print(f"FAISS index built and saved to {save_path}")

    return vectorstore


def load_vectorstore():
    """
    Loads a previously saved FAISS index from disk instead of rebuilding it
    (saves time and embedding API costs).
    """
    save_path = os.path.join(VECTORSTORE_DIR, INDEX_NAME)
    if not os.path.exists(save_path):
        raise FileNotFoundError(
            f"No saved index at {save_path}. Run `python -m src.embed_store` first."
        )

    embeddings = get_embedding_model()
    vectorstore = FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"✅ Loaded FAISS index from {save_path}")
    return vectorstore


def run_embedding_pipeline():
    """Full pipeline: ingest raw docs -> chunk -> embed -> save index."""
    chunks = run_ingestion()
    if not chunks:
        print("❌ No chunks to embed. Add files to data/raw/ first.")
        return None
    return build_vectorstore(chunks)


if __name__ == "__main__":
    run_embedding_pipeline()
"""
Ingestion module.

Responsible for: discovering files in data/raw/, loading their text content,
and splitting that content into overlapping chunks suitable for embedding.
"""
import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import RAW_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

# Maps file extensions to the LangChain loader class that knows how to read them.
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
}


def load_documents(raw_dir: str = RAW_DATA_DIR):
    """
    Walk through data/raw/, load every supported file, and return a flat
    list of LangChain Document objects (each has .page_content and .metadata).
    """
    documents = []

    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    files = os.listdir(raw_dir)
    if not files:
        print(f"No files found in {raw_dir}. Add some PDFs/.txt/.docx files first.")
        return documents

    for filename in files:
        filepath = os.path.join(raw_dir, filename)
        ext = os.path.splitext(filename)[1].lower()

        loader_class = LOADER_MAP.get(ext)
        if loader_class is None:
            print(f"Skipping unsupported file type: {filename}")
            continue

        print(f"Loading {filename}...")
        loader = loader_class(filepath)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = filename

        documents.extend(docs)

    print(f"✅ Loaded {len(documents)} raw document sections from {len(files)} files.")
    return documents


def split_documents(documents):
    """
    Split loaded documents into overlapping chunks using a splitter that
    tries paragraph breaks first, then sentences, then words.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return chunks


def run_ingestion():
    """Convenience entrypoint: load + split in one call."""
    documents = load_documents()
    if not documents:
        return []
    chunks = split_documents(documents)
    return chunks


if __name__ == "__main__":
    chunks = run_ingestion()
    if chunks:
        print("\n--- Sample chunk ---")
        print(f"Source: {chunks[0].metadata.get('source')}")
        print(f"Content: {chunks[0].page_content[:300]}...")
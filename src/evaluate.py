
"""
Evaluation script: measures retrieval precision improvement from chunking
strategy. Compares a NAIVE baseline (large chunks, no overlap, no recursive
splitting) against the OPTIMIZED setup actually used in this project
(recursive splitter, 1000 chars, 150 overlap).

Metric used: Precision@K
    Precision@K = (number of retrieved chunks that are relevant)
                  / (total number of chunks retrieved, K)

A chunk is judged "relevant" if it contains at least one of the
ground-truth keywords associated with that test question. This is a
standard, fast, reproducible way to benchmark retrieval quality without
needing a large hand-labeled dataset.

Run with: python -m src.evaluate
"""
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import OPENAI_API_KEY, EMBEDDING_MODEL, RAW_DATA_DIR, TOP_K

LOADER_MAP = {".pdf": PyPDFLoader, ".txt": TextLoader, ".docx": Docx2txtLoader}

# --- Test set: questions + the keywords that MUST appear in a truly relevant chunk ---
# Edit/expand this list to match whatever documents you put in data/raw/.
TEST_SET = [
    {
        "question": "How many remote work days are employees allowed per week?",
        "keywords": ["remote", "3 days"],
    },
    {
        "question": "How much PTO do employees accrue per year?",
        "keywords": ["18 days", "PTO"],
    },
    {
        "question": "How many sick days do employees get?",
        "keywords": ["10 paid sick days", "sick"],
    },
    {
        "question": "What is the deadline for submitting expense reports?",
        "keywords": ["30 days", "Expensify"],
    },
    {
        "question": "What is the meal reimbursement limit during travel?",
        "keywords": ["$75", "meal"],
    },
    {
        "question": "How many performance reviews happen per year and when?",
        "keywords": ["mid-year", "December", "annual review"],
    },
    {
        "question": "How much notice is required to terminate employment?",
        "keywords": ["30 days written notice"],
    },
    {
        "question": "When is the final paycheck issued after leaving the company?",
        "keywords": ["14 days", "final paycheck"],
    },
    {
        "question": "What happens if someone violates the code of conduct?",
        "keywords": ["disciplinary action", "termination"],
    },
    {
        "question": "When can a new employee start using PTO?",
        "keywords": ["90-day", "probationary"],
    },
]


def load_raw_documents():
    documents = []
    for filename in os.listdir(RAW_DATA_DIR):
        ext = os.path.splitext(filename)[1].lower()
        loader_class = LOADER_MAP.get(ext)
        if not loader_class:
            continue
        loader = loader_class(os.path.join(RAW_DATA_DIR, filename))
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = filename
        documents.extend(docs)
    return documents


def build_naive_index(documents, embeddings):
    """
    Baseline: large fixed-size chunks, NO overlap, plain character splitting
    (no recursive paragraph/sentence-aware logic). This mimics a 'quick and
    dirty' RAG setup with no chunking optimization.
    """
    splitter = CharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=0,
        separator="\n",
    )
    chunks = splitter.split_documents(documents)
    print(f"  Naive baseline: {len(chunks)} chunks (size=2000, overlap=0)")
    return FAISS.from_documents(chunks, embeddings), chunks


def build_optimized_index(documents, embeddings):
    """
    The actual project setup: recursive, semantically-aware splitting with
    overlap, as used in src/ingest.py.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"  Optimized: {len(chunks)} chunks (size=1000, overlap=150, recursive)")
    return FAISS.from_documents(chunks, embeddings), chunks


def is_relevant(chunk_text: str, keywords: list[str]) -> bool:
    """A chunk counts as relevant if ANY of the question's keywords appear in it."""
    text_lower = chunk_text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def evaluate_index(vectorstore, label: str):
    """
    Runs the full test set against a given vectorstore, computes Precision@K
    for each question, and returns the average across all questions.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    precisions = []

    print(f"\n--- Evaluating: {label} ---")
    for item in TEST_SET:
        question = item["question"]
        keywords = item["keywords"]

        results = retriever.invoke(question)
        relevant_count = sum(1 for doc in results if is_relevant(doc.page_content, keywords))
        precision = relevant_count / len(results) if results else 0
        precisions.append(precision)

        print(f"  [{precision:.2f}] {question}")

    avg_precision = sum(precisions) / len(precisions)
    print(f"  → Average Precision@{TOP_K} for {label}: {avg_precision:.3f}")
    return avg_precision


def run_evaluation():
    print("📥 Loading raw documents...")
    documents = load_raw_documents()
    if not documents:
        print("❌ No documents found in data/raw/. Add files first.")
        return

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)

    print("\n🔧 Building indexes...")
    naive_store, naive_chunks = build_naive_index(documents, embeddings)
    optimized_store, optimized_chunks = build_optimized_index(documents, embeddings)

    naive_precision = evaluate_index(naive_store, "NAIVE baseline (size=2000, no overlap)")
    optimized_precision = evaluate_index(optimized_store, "OPTIMIZED (recursive, size=1000, overlap=150)")

    if naive_precision > 0:
        improvement = ((optimized_precision - naive_precision) / naive_precision) * 100
    else:
        improvement = float("inf")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Naive baseline Precision@{TOP_K}:     {naive_precision:.3f}")
    print(f"Optimized setup Precision@{TOP_K}:    {optimized_precision:.3f}")
    print(f"Relative improvement:           {improvement:+.1f}%")
    print("=" * 60)
    print(f"\nInterpretation: irrelevant chunks retrieved dropped from "
          f"{(1-naive_precision)*100:.1f}% to {(1-optimized_precision)*100:.1f}% "
          f"of all retrieved results.")


if __name__ == "__main__":
    run_evaluation()

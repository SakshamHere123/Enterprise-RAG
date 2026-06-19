"""
Retriever module.

Responsible for: taking a vectorstore and exposing a simple "given a query,
return the top-K most relevant chunks" interface.
"""
from src.embed_store import load_vectorstore
from src.config import TOP_K


def get_retriever(k: int = TOP_K, search_type: str = "similarity"):
    """
    Returns a LangChain retriever built on our FAISS index.

    search_type options:
      - "similarity": plain nearest-neighbor search (default)
      - "mmr": balances relevance with diversity, useful if docs have a lot
               of near-duplicate content
    """
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )
    return retriever


def retrieve_chunks(query: str, k: int = TOP_K):
    """Direct utility: given a query string, return the top-k matching chunks."""
    retriever = get_retriever(k=k)
    results = retriever.invoke(query)
    return results


if __name__ == "__main__":
    test_query = input("Enter a test query: ")
    results = retrieve_chunks(test_query)
    print(f"\n🔍 Top {len(results)} matches for: \"{test_query}\"\n")
    for i, doc in enumerate(results, 1):
        print(f"--- Match {i} (source: {doc.metadata.get('source')}) ---")
        print(doc.page_content[:300])
        print()
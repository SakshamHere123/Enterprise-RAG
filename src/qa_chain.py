"""
QA Chain module — the heart of the RAG system.

Responsible for: retrieving relevant chunks, stuffing them into a prompt
template, and calling the LLM to generate a grounded, hallucination-resistant
answer.
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.config import OPENAI_API_KEY, CHAT_MODEL
from src.retriever import get_retriever

PROMPT_TEMPLATE = """You are a knowledgeable enterprise assistant. Answer the
user's question using ONLY the context provided below. 

Rules:
- If the context does not contain enough information to answer, say so
  honestly instead of guessing or using outside knowledge.
- Be concise and precise — this is for technical/business use, not casual chat.
- When relevant, mention which source document(s) the information came from.

Context:
{context}

Question: {question}

Answer:"""


def format_docs(docs):
    """Merges retrieved chunks into a single context string, labeled by source."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_qa_chain():
    """
    Builds the full RAG chain using LangChain Expression Language (LCEL).
    The pipe (|) operator chains steps: retriever -> prompt -> llm -> parser.
    """
    retriever = get_retriever()
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0,  # 0 = deterministic, factual answers
    )
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def ask(question: str) -> str:
    """Convenience function: ask a question, get an answer string back."""
    chain = build_qa_chain()
    return chain.invoke(question)


def ask_with_sources(question: str):
    """
    Same as ask(), but also returns retrieved chunks/sources — used by the
    API/frontend to show 'sources used' for trust and traceability.
    """
    retriever = get_retriever()
    docs = retriever.invoke(question)
    context = format_docs(docs)

    llm = ChatOpenAI(model=CHAT_MODEL, openai_api_key=OPENAI_API_KEY, temperature=0)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    parser = StrOutputParser()

    chain = prompt | llm | parser
    answer = chain.invoke({"context": context, "question": question})

    sources = list({doc.metadata.get("source", "unknown") for doc in docs})
    return {"answer": answer, "sources": sources, "chunks_used": len(docs)}


if __name__ == "__main__":
    q = input("Ask a question about your documents: ")
    result = ask_with_sources(q)
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources: {', '.join(result['sources'])}")
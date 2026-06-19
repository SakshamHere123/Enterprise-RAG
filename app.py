"""
FastAPI backend.

Exposes the RAG pipeline as a web API so any frontend can talk to it over HTTP.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.qa_chain import ask_with_sources
from src.embed_store import run_embedding_pipeline

app = FastAPI(title="Enterprise RAG API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Enterprise RAG API is running"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = ask_with_sources(request.question)
        return result
    except FileNotFoundError:
        raise HTTPException(
            status_code=400,
            detail="No vector index found. Run /reindex first.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reindex")
def reindex():
    try:
        vectorstore = run_embedding_pipeline()
        if vectorstore is None:
            raise HTTPException(status_code=400, detail="No documents found in data/raw/")
        return {"status": "ok", "message": "Reindexing complete"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
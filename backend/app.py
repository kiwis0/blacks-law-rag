
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag import index_document, query_rag, chunks, metadata  # Fixed import
import os

app = FastAPI(title="Black's Law Dictionary RAG")

PDF_PATH = os.path.join("content", "aff.pdf")

class QueryRequest(BaseModel):
    question: str

class LookupRequest(BaseModel):
    term: str

@app.post("/index")
async def index_pdf():
    if not os.path.exists(PDF_PATH):
        raise HTTPException(status_code=404, detail="PDF not found")
    index_document(PDF_PATH)
    return {"message": "Dictionary indexed successfully", "chunk_count": len(chunks)}

@app.post("/query")
async def query(request: QueryRequest):
    answer = query_rag(request.question)
    return {"question": request.question, "answer": answer}

@app.post("/lookup")
async def lookup(request: LookupRequest):
    term = request.term.upper().strip()
    print(f"Looking up term: {term}")
    print(f"Metadata: {metadata}")
    for chunk, meta in zip(chunks, metadata):
        if meta == term:
            return {"term": term, "definition": chunk.split(":", 1)[1].strip()}
    raise HTTPException(status_code=404, detail=f"Term '{term}' not found")
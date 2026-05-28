"""
PRJ-04: Corrective RAG — Corrective RAG with Self-Reflection
Especializado para contexto jurídico.
"""

import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from .core.corrective_engine import CorrectiveRAG
from .api.schemas import QueryRequest, QueryResponse, HealthResponse

load_dotenv()

app = FastAPI(
    title="PRJ-04: Corrective RAG",
    description="Pipeline RAG auto-corrente com self-reflection para contexto jurídico",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_engine = None

@app.on_event("startup")
async def startup():
    global rag_engine
    rag_engine = CorrectiveRAG()

@app.get("/", response_model=HealthResponse)
async def root():
    return {
        "status": "ready",
        "engine": "CorrectiveRAG",
        "model": os.getenv("MODEL_NAME", "llama3.2:3b"),
        "features": ["self_reflection", "retry_loop", "quality_score"]
    }

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if rag_engine is None:
        raise HTTPException(503, "Engine not initialized")
    
    try:
        result = rag_engine.query(
            question=request.question,
            max_retries=request.max_retries or 2
        )
        return result
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/upload_pdf", status_code=201)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDFs allowed")
    
    try:
        os.makedirs(rag_engine.uploads_dir, exist_ok=True)
        path = os.path.join(rag_engine.uploads_dir, file.filename)
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        chunks = rag_engine.process_pdf(path)
        return {"filename": file.filename, "chunks": chunks}
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
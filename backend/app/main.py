"""
EmbeddingFlow Backend - FastAPI app.
Phase 1, Task 1.1 only: Query Analysis.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import Settings
from app.phases.phase1 import analyze_query, QueryAnalysisResult

app = FastAPI(
    title="EmbeddingFlow API",
    description="Phase 1 Task 1.1: Query Analysis only",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = Settings()


class AnalyzeQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User search query")


@app.post("/api/v1/analyze-query", response_model=QueryAnalysisResult)
def api_analyze_query(body: AnalyzeQueryRequest):
    """
    Task 1.1: Query Analysis only.
    Returns: intent, entities, time_sensitive, complexity.
    """
    try:
        return analyze_query(body.query, settings=settings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}

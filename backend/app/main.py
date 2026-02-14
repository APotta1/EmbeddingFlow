"""
EmbeddingFlow API — Phase 1 Task 1.1 Query Analysis endpoint.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.phases.phase1 import analyze_query
from app.phases.phase1.schemas import QueryAnalysisResponse

app = FastAPI(title="EmbeddingFlow", version="0.1.0")


class AnalyzeRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/query/analyze", response_model=QueryAnalysisResponse)
def query_analyze(body: AnalyzeRequest):
    """
    Phase 1, Task 1.1: Query Analysis.

    Returns structured query_analysis (intent, entities, time_sensitive, complexity) for Phase 2.
    """
    try:
        return analyze_query(body.query)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

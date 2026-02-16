"""
EmbeddingFlow API — Phase 1: Query Analysis (1.1), Decomposition (1.2), Expansion (1.3).
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.phases.phase1 import analyze_query, run_phase1
from app.phases.phase1.pipeline import to_phase2_payload
from app.phases.phase1.schemas import Phase2Payload, QueryAnalysisResponse

app = FastAPI(title="EmbeddingFlow", version="0.1.0")


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/query/analyze", response_model=QueryAnalysisResponse)
def query_analyze(body: QueryRequest):
    """
    Phase 1, Task 1.1 only: Query Analysis (intent, entities, time_sensitive, complexity).
    """
    try:
        return analyze_query(body.query)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/api/v1/query/process", response_model=Phase2Payload)
def query_process(body: QueryRequest):
    """
    Phase 1 full pipeline: 1.1 → 1.2 → 1.3, then return JSON in Phase 2 format.

    Returns: original_query, intent, entities, time_sensitivity, subqueries, search_variants, constraints.
    """
    try:
        phase1 = run_phase1(body.query)
        return to_phase2_payload(body.query, phase1)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

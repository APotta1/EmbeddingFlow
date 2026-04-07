"""EmbeddingFlow API endpoints."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.phases.phase1 import analyze_query, run_phase1
from app.phases.phase1.pipeline import to_phase2_payload
from app.phases.phase1.schemas import Phase2Payload, QueryAnalysisResponse
from app.phases.phase4.schemas import Phase4Output
from app.phases.phase5.pipeline import run_phase5
from app.phases.phase5.schemas import Phase5Config, Phase5Output

app = FastAPI(title="EmbeddingFlow", version="0.1.0")


class QueryRequest(BaseModel):
    query: str


class PhaseInfo(BaseModel):
    id: int
    name: str
    description: str


class Phase5Request(BaseModel):
    phase4_output: Phase4Output
    config: Phase5Config | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/v1/phases", response_model=list[PhaseInfo])
def list_phases():
    """
    Return all currently implemented pipeline phases.
    Useful for phase selection/dropdown UIs.
    """
    return [
        PhaseInfo(id=1, name="Query Processing", description="Analyze, decompose, expand query."),
        PhaseInfo(id=2, name="Web Search & Retrieval", description="Search, dedupe, and rank URLs."),
        PhaseInfo(id=3, name="Content Extraction & Cleaning", description="Fetch and clean document text."),
        PhaseInfo(id=4, name="Contextual Chunking", description="Chunk documents and enrich chunk context."),
        PhaseInfo(id=5, name="Embedding & Indexing", description="Generate embeddings and store vectors."),
    ]


@app.post("/api/v1/query/analyze", response_model=QueryAnalysisResponse)
def query_analyze(body: QueryRequest):
    """
    Phase 1, Task 1.1 only: Query Analysis (intent, entities, time_sensitive, complexity).
    """
    try:
        return analyze_query(body.query)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/api/v1/phase5/run", response_model=Phase5Output)
def phase5_run(body: Phase5Request):
    """
    Run Phase 5 on top of an existing Phase4Output payload.
    """
    try:
        return run_phase5(body.phase4_output, config=body.config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Phase 5 failed: {str(e)}")


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

"""
Phase 1 pipeline: run Task 1.1 → 1.2 → 1.3 and return output for Phase 2.
"""

from .query_analysis import analyze_query
from .query_decomposition import decompose_query
from .query_expansion import expand_query
from .schemas import (
    Phase1Output,
    Phase2Payload,
    TimeSensitivityPayload,
)


def run_phase1(query: str) -> Phase1Output:
    """
    Run full Phase 1 query processing: analysis → decomposition → expansion.

    Returns internal Phase1Output. Use to_phase2_payload() to get JSON for Phase 2.
    """
    analysis_response = analyze_query(query)
    query_analysis = analysis_response.query_analysis

    decomposition = decompose_query(query, query_analysis)
    expansion = expand_query(query, query_analysis)

    return Phase1Output(
        query_analysis=query_analysis,
        query_decomposition=decomposition,
        query_expansion=expansion,
    )


def to_phase2_payload(original_query: str, phase1: Phase1Output) -> Phase2Payload:
    """
    Convert Phase 1 output to the JSON format sent to Phase 2 (Web Search & Retrieval).

    Example output shape:
    {
      "original_query": "How does the Federal Reserve control inflation?",
      "intent": "explanatory",
      "entities": [{"name": "Federal Reserve", "type": "organization"}, ...],
      "time_sensitivity": {"is_time_sensitive": false, "date_range": null},
      "subqueries": ["What tools does the Federal Reserve use to control inflation?", ...],
      "search_variants": ["tools does the Federal Reserve use to control inflation?", ...],
      "constraints": {"source_types": ["news", "academic", "government"], "max_results_per_query": 10, "language": "en"}
    }
    """
    qa = phase1.query_analysis
    return Phase2Payload(
        original_query=original_query.strip(),
        intent=qa.intent.primary,
        entities=[{"name": e.text, "type": e.type} for e in qa.entities],
        time_sensitivity=TimeSensitivityPayload(
            is_time_sensitive=qa.time_sensitive,
            date_range=qa.time_expressions if qa.time_sensitive else None,
        ),
        subqueries=phase1.query_decomposition.sub_questions,
        search_variants=phase1.query_expansion.search_variations,
    )

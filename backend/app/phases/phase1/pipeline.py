"""
Phase 1 pipeline: run Task 1.1 → 1.2 → 1.3 and return combined output for Phase 2.
"""

from .query_analysis import analyze_query
from .query_decomposition import decompose_query
from .query_expansion import expand_query
from .schemas import Phase1Output


def run_phase1(query: str) -> Phase1Output:
    """
    Run full Phase 1 query processing: analysis → decomposition → expansion.

    Returns combined output ready to send to Phase 2 (Web Search & Retrieval).
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

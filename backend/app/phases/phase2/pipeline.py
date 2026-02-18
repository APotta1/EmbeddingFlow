"""
Phase 2 pipeline: run payload → search → rank and return output for Phase 3.

Orchestrates: query optimization → parallel search (Tavily + Serper) →
merge/dedupe → rank and filter (Groq domain credibility, top N).
Single entry point: run_phase2(phase2_payload).
"""

from app.phases.phase1.schemas import Phase2Payload
from app.phases.phase2.ranking import rank_and_select
from app.phases.phase2.schemas import Phase2Output
from app.phases.phase2.search_orchestrator import search_parallel


def run_phase2(
    payload: Phase2Payload,
    top_n: int = 20,
    *,
    min_results_per_source: int = 0,
) -> Phase2Output:
    """
    Run full Phase 2: optimize queries → parallel search (both APIs) →
    merge/dedupe → rank and filter → return top N URLs for Phase 3.

    Input: Phase2Payload (from Phase 1 or API).
    Output: Phase2Output with original_query, top 15–20 ranked URLs,
    total_searched, and queries_used.

    Args:
        payload: Phase2Payload with original_query, subqueries, search_variants, etc.
        top_n: Number of URLs to return after ranking (default 20).
        min_results_per_source: If > 0, ensure at least this many from each API in top N (e.g. 2).

    Returns:
        Phase2Output ready for Phase 3 (content extraction).
    """
    # Search: optimize queries, run Tavily + Serper in parallel, merge and dedupe
    search_output = search_parallel(payload)

    if not search_output.urls:
        return Phase2Output(
            original_query=payload.original_query,
            urls=[],
            total_searched=search_output.total_searched,
            queries_used=search_output.queries_used,
        )

    # Rank: filter off-topic, then position + domain + recency + relevance; HTTPS only
    ranked_urls = rank_and_select(
        search_output.urls,
        top_n=top_n,
        time_sensitive=payload.time_sensitivity.is_time_sensitive,
        original_query=payload.original_query,
        min_results_per_source=min_results_per_source,
    )

    return Phase2Output(
        original_query=payload.original_query,
        urls=ranked_urls,
        total_searched=search_output.total_searched,
        queries_used=search_output.queries_used,
    )

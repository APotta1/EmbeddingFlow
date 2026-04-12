"""
Phase 1 pipeline.

The two Groq chat completions (Meta Llama) for query analysis and retrieval strategy are
submitted together and run concurrently, then results are merged (strategy clamped by
analysis complexity). No other phases are involved in that parallelism.
"""

from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor

from .query_analysis import analyze_query
from .query_retrieval_strategy import (
    clamp_retrieval_plan,
    engine_queries_for_complexity,
    plan_retrieval_strategy,
)
from .schemas import (
    ComplexityOutput,
    IntentOutput,
    Phase1Output,
    Phase2Payload,
    QueryAnalysisOutput,
    QueryRetrievalPlanOutput,
    TimeSensitivityPayload,
)


def _fallback_analysis() -> QueryAnalysisOutput:
    return QueryAnalysisOutput(
        intent=IntentOutput(primary="other", categories=[]),
        entities=[],
        time_sensitive=False,
        time_expressions=[],
        complexity=ComplexityOutput(level="simple", suggested_sub_questions=0, multi_hop=False),
    )


def run_phase1(query: str) -> Phase1Output:
    """
    Run Phase 1: fire two Meta Llama (Groq) requests at once—query analysis and retrieval
    strategy—wait for both, then clamp the strategy using ``analysis.complexity``.

    Returns internal Phase1Output. Use to_phase2_payload() to get JSON for Phase 2.
    """
    query = (query or "").strip()
    if not query:
        return Phase1Output(
            query_analysis=_fallback_analysis(),
            query_retrieval_plan=QueryRetrievalPlanOutput(),
        )

    def _analysis() -> QueryAnalysisOutput:
        return analyze_query(query).query_analysis

    def _strategy() -> QueryRetrievalPlanOutput:
        return plan_retrieval_strategy(query)

    # Both submits return immediately; worker threads run two Groq completions in parallel.
    with ThreadPoolExecutor(max_workers=2) as pool:
        fa = pool.submit(_analysis)
        fs = pool.submit(_strategy)
        try:
            query_analysis = fa.result()
        except Exception as e:
            print(f"Phase 1 query_analysis failed: {e}")
            traceback.print_exc()
            query_analysis = _fallback_analysis()
        try:
            raw_plan = fs.result()
        except Exception as e:
            print(f"Phase 1 retrieval strategy failed: {e}")
            traceback.print_exc()
            raw_plan = QueryRetrievalPlanOutput(hyde_document=query)

    plan = clamp_retrieval_plan(query, query_analysis.complexity.level, raw_plan)
    return Phase1Output(
        query_analysis=query_analysis,
        query_retrieval_plan=plan,
    )


def to_phase2_payload(original_query: str, phase1: Phase1Output) -> Phase2Payload:
    """
    Convert Phase 1 output to the JSON format sent to Phase 2 (Web Search & Retrieval).
    Fills engine-specific Tavily/Serper lists from HyDE + keyword variants + sub-questions.
    """
    oq = original_query.strip()
    qa = phase1.query_analysis
    plan = phase1.query_retrieval_plan
    level = qa.complexity.level.lower()

    tavily_queries, serper_queries = engine_queries_for_complexity(
        level,
        plan,
        oq,
        time_sensitive=qa.time_sensitive,
    )
    hyde = (plan.hyde_document or "").strip() or oq

    subqueries = list(plan.sub_questions) if plan.sub_questions else []

    keyword_for_payload = plan.keyword_variants or [hyde]

    return Phase2Payload(
        original_query=oq,
        intent=qa.intent.primary,
        entities=[{"name": e.text, "type": e.type} for e in qa.entities],
        time_sensitivity=TimeSensitivityPayload(
            is_time_sensitive=qa.time_sensitive,
            date_range=qa.time_expressions if qa.time_sensitive else None,
        ),
        subqueries=subqueries,
        search_variants=keyword_for_payload,
        hyde_document=hyde,
        tavily_queries=tavily_queries,
        serper_queries=serper_queries,
    )

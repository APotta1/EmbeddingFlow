"""
Tavily Search API client. Uses TAV_API_KEY.
"""

import os
from typing import Optional
from urllib.parse import urlparse

from tavily import TavilyClient

from app.phases.phase1.schemas import Phase2Payload
from app.phases.phase2.schemas import SearchResult
from app.phases.phase2.support import get_performance_monitor, get_rate_limiter, get_search_cache

# Reuse client for performance (avoids repeated client setup)
_tavily_client: Optional[TavilyClient] = None


def _get_client() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        api_key = os.environ.get("TAV_API_KEY")
        if not api_key:
            raise ValueError("TAV_API_KEY environment variable is required")
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


def search_tavily(
    query: str,
    payload: Phase2Payload,
    max_results: Optional[int] = None,
    use_cache: bool = True,
) -> list[SearchResult]:
    api_name = "tavily"
    cache = get_search_cache()
    monitor = get_performance_monitor()
    rate_limiter = get_rate_limiter(api_name)
    metric = monitor.start_search(query, api_name)

    if use_cache:
        cached = cache.get(query, api_name)
        if cached:
            monitor.record_success(metric, len(cached), cache_hit=True)
            return cached

    rate_limiter.wait_if_needed(api_name)

    try:
        client = _get_client()
        max_results = max_results or payload.constraints.max_results_per_query
        # Use "advanced" for definitional/how-to/factual so Tavily returns more on-topic results
        search_depth = (
            "advanced"
            if payload.intent in ["explanatory", "comparison", "how_to", "factual"]
            else "basic"
        )
        topic = "news" if (payload.intent == "news" or "news" in payload.constraints.source_types) else "general"
        search_params = {
            "query": query,
            "search_depth": search_depth,
            "topic": topic,
            "max_results": min(max_results, 20),
            "include_answer": True,  # encourages results that directly address the query
        }
        if payload.time_sensitivity.is_time_sensitive:
            search_params["time_range"] = "month"

        response = client.search(**search_params)
        results = []
        if "results" in response:
            for idx, item in enumerate(response["results"], start=1):
                domain = None
                try:
                    domain = urlparse(item.get("url", "")).netloc
                except Exception:
                    pass
                results.append(
                    SearchResult(
                        url=item.get("url", ""),
                        title=item.get("title", ""),
                        snippet=item.get("content", ""),
                        source_api="tavily",
                        position=idx,
                        domain=domain,
                        published_date=item.get("published_date"),
                    )
                )

        rate_limiter.record_success(api_name)
        monitor.record_success(metric, len(results), cache_hit=False)
        if use_cache and results:
            cache.set(query, api_name, results)
        return results

    except Exception as e:
        is_rate_limit = "429" in str(e) or "rate limit" in str(e).lower()
        rate_limiter.record_error(api_name, is_rate_limit_error=is_rate_limit)
        monitor.record_error(metric, str(e))
        print(f"Tavily search error for query '{query}': {e}")
        return []

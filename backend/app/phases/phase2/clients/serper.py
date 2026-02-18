"""
Serper Search API client. Uses SERP_API_KEY. Reuses a single requests.Session for performance.
"""

import os
from typing import Optional
from urllib.parse import urlparse

import requests

from app.phases.phase1.schemas import Phase2Payload
from app.phases.phase2.schemas import SearchResult
from app.phases.phase2.support import get_performance_monitor, get_rate_limiter, get_search_cache

# Reuse session for connection pooling and lower latency
_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


def _get_api_key() -> str:
    api_key = os.environ.get("SERP_API_KEY")
    if not api_key:
        raise ValueError("SERP_API_KEY environment variable is required")
    return api_key


def search_serper(
    query: str,
    payload: Phase2Payload,
    max_results: Optional[int] = None,
    use_cache: bool = True,
) -> list[SearchResult]:
    api_name = "serper"
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
        api_key = _get_api_key()
        max_results = max_results or payload.constraints.max_results_per_query
        request_data = {"q": query, "num": min(max_results, 100)}
        if payload.constraints.language != "en":
            request_data["gl"] = payload.constraints.language
        if payload.time_sensitivity.is_time_sensitive:
            request_data["tbs"] = "qdr:m"
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

        response = _get_session().post(
            "https://google.serper.dev/search",
            json=request_data,
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for idx, item in enumerate(data.get("organic", []), start=1):
            domain = None
            try:
                domain = urlparse(item.get("link", "")).netloc
            except Exception:
                pass
            results.append(
                SearchResult(
                    url=item.get("link", ""),
                    title=item.get("title", ""),
                    snippet=item.get("snippet", ""),
                    source_api="serper",
                    position=idx,
                    domain=domain,
                    published_date=item.get("date"),
                )
            )

        rate_limiter.record_success(api_name)
        monitor.record_success(metric, len(results), cache_hit=False)
        if use_cache and results:
            cache.set(query, api_name, results)
        return results

    except requests.exceptions.HTTPError as e:
        is_rate_limit = e.response.status_code == 429
        rate_limiter.record_error(api_name, is_rate_limit_error=is_rate_limit)
        monitor.record_error(metric, f"HTTP {e.response.status_code}: {str(e)}")
        print(f"Serper search error for query '{query}': {e}")
        return []
    except requests.exceptions.RequestException as e:
        rate_limiter.record_error(api_name, is_rate_limit_error=False)
        monitor.record_error(metric, str(e))
        print(f"Serper search error for query '{query}': {e}")
        return []
    except Exception as e:
        rate_limiter.record_error(api_name, is_rate_limit_error=False)
        monitor.record_error(metric, str(e))
        print(f"Serper search error for query '{query}': {e}")
        return []

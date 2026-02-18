"""
Search orchestrator: parallel search across both APIs and all optimized queries.
"""

import concurrent.futures
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from app.phases.phase1.schemas import Phase2Payload
from app.phases.phase2.clients import search_serper, search_tavily
from app.phases.phase2.query_optimizer import optimize_queries
from app.phases.phase2.schemas import Phase2Output, SearchResult

# Always use both APIs in parallel
SEARCH_APIS = ["tavily", "serper"]


@dataclass
class SearchTask:
    query: str
    api_name: str
    payload: Phase2Payload
    max_results: Optional[int] = None


def _relevance_score(result: SearchResult, original_query: str) -> float:
    """Score by query-term coverage; title matches count more than snippet (accuracy)."""
    words = [w for w in re.findall(r"\w+", original_query.lower()) if len(w) > 1]
    if not words:
        return 0.0
    title_lower = (result.title or "").lower()
    snippet_lower = (result.snippet or "").lower()
    score = 0.0
    for w in words:
        if w in title_lower:
            score += 1.5  # title match = stronger signal
        elif w in snippet_lower:
            score += 1.0
    return score / len(words)


def _deduplicate_results(results: list[SearchResult]) -> list[SearchResult]:
    seen_urls = set()
    deduplicated = []
    for result in results:
        url_lower = result.url.lower().strip()
        if url_lower and url_lower not in seen_urls:
            seen_urls.add(url_lower)
            result.position = len(deduplicated) + 1
            deduplicated.append(result)
    return deduplicated


def _merge_results(
    all_results: list[list[SearchResult]],
    original_query: str,
) -> list[SearchResult]:
    """Merge by URL, pick best result per URL, then sort by relevance and position."""
    url_to_results: dict[str, list[SearchResult]] = defaultdict(list)
    for results in all_results:
        for result in results:
            url_lower = result.url.lower().strip()
            if url_lower:
                url_to_results[url_lower].append(result)

    merged = []
    for url, candidates in url_to_results.items():
        if not candidates:
            continue
        # Prefer longer snippet, then Serper, then lower position; then boost by query relevance
        best = max(
            candidates,
            key=lambda r: (
                len(r.snippet or ""),
                1 if r.source_api == "serper" else 0,
                -r.position,
            ),
        )
        merged.append(best)

    # Sort by relevance (accuracy), then API, then position
    merged.sort(
        key=lambda r: (
            -_relevance_score(r, original_query),
            0 if r.source_api == "serper" else 1,
            r.position,
        )
    )

    # Domain diversity: avoid clustering same domain (improves result variety/accuracy)
    merged = _apply_domain_diversity(merged)

    for idx, result in enumerate(merged, start=1):
        result.position = idx
    return merged


def _apply_domain_diversity(results: list[SearchResult], window: int = 3) -> list[SearchResult]:
    """Reorder so the same domain does not dominate consecutive positions."""
    if len(results) <= window:
        return results
    out: list[SearchResult] = []
    used: set[int] = set()
    recent_domains: list[str] = []
    while len(out) < len(results):
        best_idx = -1
        best_penalty = float("inf")
        for i, r in enumerate(results):
            if i in used:
                continue
            domain = (r.domain or "").lower() or r.url
            penalty = recent_domains.count(domain)
            if penalty < best_penalty:
                best_penalty = penalty
                best_idx = i
        if best_idx < 0:
            break
        used.add(best_idx)
        r = results[best_idx]
        out.append(r)
        recent_domains.append((r.domain or "").lower() or r.url)
        if len(recent_domains) > window:
            recent_domains.pop(0)
    return out


def _execute_search_task(task: SearchTask) -> list[SearchResult]:
    if task.api_name == "tavily":
        return search_tavily(
            query=task.query,
            payload=task.payload,
            max_results=task.max_results,
            use_cache=True,
        )
    if task.api_name == "serper":
        return search_serper(
            query=task.query,
            payload=task.payload,
            max_results=task.max_results,
            use_cache=True,
        )
    return []


def search_parallel(
    payload: Phase2Payload,
    max_workers: Optional[int] = None,
    max_results_per_query: Optional[int] = None,
) -> Phase2Output:
    """Run parallel search across both APIs and all optimized queries."""
    optimized = optimize_queries(payload)
    queries = optimized.queries

    if not queries:
        return Phase2Output(
            original_query=payload.original_query,
            urls=[],
            total_searched=0,
            queries_used=[],
        )

    tasks = []
    max_results = max_results_per_query or payload.constraints.max_results_per_query
    for query in queries:
        for api_name in SEARCH_APIS:
            tasks.append(
                SearchTask(query=query, api_name=api_name, payload=payload, max_results=max_results)
            )

    max_workers = max_workers if max_workers is not None else min(10, len(tasks))
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(_execute_search_task, task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                all_results.append(future.result())
            except Exception as e:
                print(f"Error executing search task {task.query} ({task.api_name}): {e}")
                all_results.append([])

    merged = _merge_results(all_results, payload.original_query)
    final_results = _deduplicate_results(merged)
    max_total_results = max_results * len(queries)
    if len(final_results) > max_total_results:
        final_results = final_results[:max_total_results]

    return Phase2Output(
        original_query=payload.original_query,
        urls=final_results,
        total_searched=len(tasks),
        queries_used=queries,
    )

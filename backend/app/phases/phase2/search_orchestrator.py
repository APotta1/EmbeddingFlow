"""
Search orchestrator: parallel search across both APIs and all optimized queries.
"""

#parrallel search across both APIs and all optimized queries
#sequential search is being used to search the web for information one by one

import concurrent.futures
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from app.phases.phase1.schemas import Phase2Payload
from app.phases.phase2.bm25 import bm25_score, filter_by_bm25, tokenize
from app.phases.phase2.clients import search_serper, search_tavily
from app.phases.phase2.query_optimizer import optimize_queries
from app.phases.phase2.schemas import Phase2Output, SearchResult

# Always use both APIs in parallel
SEARCH_APIS = ["tavily", "serper"]
BLOCKED_DOMAINS = {
    "www.youtube.com",
    "youtube.com",
    "youtu.be",
    "www.reddit.com",
    "reddit.com",
    "old.reddit.com",
    "www.quora.com",
    "quora.com",
    "www.facebook.com",
    "facebook.com",
    "m.facebook.com",
    "www.instagram.com",
    "instagram.com",
    "www.twitter.com",
    "twitter.com",
    "x.com",
    "www.x.com",
    "mobile.twitter.com",
    "t.co",
    "istockphoto.com",
    "www.istockphoto.com",
    "shutterstock.com",
    "www.shutterstock.com",
    "gettyimages.com",
    "www.gettyimages.com",
    "alamy.com",
    "www.alamy.com",
    "depositphotos.com",
    "www.depositphotos.com",
    "brainly.com",
    "www.brainly.com",
    "brainly.in",
    "www.brainly.in",
    "brainly.co.id",
    "www.brainly.co.id",
    "chegg.com",
    "www.chegg.com",
    "coursehero.com",
    "www.coursehero.com",
    "answers.com",
    "www.answers.com",
    "ask.com",
    "www.ask.com",
    "reference.com",
    "www.reference.com",
    "sciencedirect.com",
    "www.sciencedirect.com",
    "jstor.org",
    "www.jstor.org",
    "link.springer.com",
    "www.link.springer.com",
    "onlinelibrary.wiley.com",
    "www.onlinelibrary.wiley.com",
    "altoida.com",
    "www.altoida.com",
}


@dataclass
class SearchTask:
    query: str
    api_name: str
    payload: Phase2Payload
    max_results: Optional[int] = None


def _is_blocked(result: SearchResult) -> bool:
    domain = (result.domain or "").lower().strip()
    return domain in BLOCKED_DOMAINS


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
    """Merge by URL, then rank/filter via BM25 and apply domain diversity."""
    filtered_results = [
        [result for result in results if not _is_blocked(result)] for results in all_results
    ]
    url_to_results: dict[str, list[SearchResult]] = defaultdict(list)
    for results in filtered_results:
        for result in results:
            url_lower = result.url.lower().strip()
            if url_lower:
                url_to_results[url_lower].append(result)

    merged = []
    for url, candidates in url_to_results.items():
        if not candidates:
            continue
        # Prefer longer snippet, then Tavily (typically higher-signal for merged factual retrieval)
        best = max(
            candidates,
            key=lambda r: (
                len(r.snippet or ""),
                1 if r.source_api == "tavily" else 0,
                -r.position,
            ),
        )
        merged.append(best)

    n_docs = len(merged)
    doc_freqs: dict[str, int] = defaultdict(int)
    doc_lengths: dict[str, int] = {}
    for result in merged:
        doc_key = result.url.lower().strip()
        tokens = tokenize(f"{result.title or ''} {result.snippet or ''}")
        doc_lengths[doc_key] = len(tokens)
        for term in set(tokens):
            doc_freqs[term] += 1

    bm25_scores: dict[str, float] = {}
    for result in merged:
        url_lower = result.url.lower().strip()
        bm25_scores[url_lower] = bm25_score(
            original_query,
            result.title or "",
            result.snippet or "",
            result.url.lower().strip(),
            doc_lengths,
            doc_freqs,
            n_docs,
        )

    merged.sort(
        key=lambda r: (
            -bm25_scores.get(r.url.lower().strip(), 0.0),
            0 if r.source_api == "tavily" else 1,
            r.position,
        )
    )

    merged = filter_by_bm25(merged, original_query)

    for idx, result in enumerate(merged, start=1):
        result.position = idx

    # Domain diversity runs after BM25 filtering and positional reassignment.
    merged = _apply_domain_diversity(merged)
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
    """Run parallel search across Tavily and Serper using engine-specific queries when set."""
    max_results = max_results_per_query or payload.constraints.max_results_per_query

    tavily_queries = [q.strip() for q in (payload.tavily_queries or []) if q and q.strip()]
    serper_queries = [q.strip() for q in (payload.serper_queries or []) if q and q.strip()]

    if tavily_queries or serper_queries:
        tasks: list[SearchTask] = []
        for query in tavily_queries:
            tasks.append(
                SearchTask(
                    query=query,
                    api_name="tavily",
                    payload=payload,
                    max_results=max_results,
                )
            )
        for query in serper_queries:
            tasks.append(
                SearchTask(
                    query=query,
                    api_name="serper",
                    payload=payload,
                    max_results=max_results,
                )
            )
        queries_used = list(dict.fromkeys(tavily_queries + serper_queries))
    else:
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
        for query in queries:
            for api_name in SEARCH_APIS:
                tasks.append(
                    SearchTask(
                        query=query,
                        api_name=api_name,
                        payload=payload,
                        max_results=max_results,
                    )
                )
        queries_used = queries

    max_workers = max_workers if max_workers is not None else min(10, max(1, len(tasks)))
    all_results: list[list[SearchResult]] = []
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

    return Phase2Output(
        original_query=payload.original_query,
        urls=final_results,
        total_searched=len(tasks),
        queries_used=queries_used,
    )

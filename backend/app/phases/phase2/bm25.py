import math
import re
from collections import Counter, defaultdict
from statistics import mean

from app.phases.phase2.schemas import SearchResult

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\w+", (text or "").lower())
    return [t for t in tokens if len(t) > 2 and t not in STOP_WORDS]


def bm25_score(
    query: str,
    title: str,
    snippet: str,
    url: str,
    doc_lengths: dict[str, int],
    doc_freqs: dict[str, int],
    n_docs: int,
    k1: float = 1.5,
    b: float = 0.75,
    title_weight: float = 2.0,
) -> float:
    if n_docs <= 0:
        return 0.0
    query_terms = tokenize(query)
    if not query_terms:
        return 0.0

    title_tokens = tokenize(title)
    boosted_title_tokens = title_tokens * int(title_weight)
    doc_tokens = boosted_title_tokens + tokenize(snippet)
    if not doc_tokens:
        return 0.0

    tf_counts = Counter(doc_tokens)
    doc_key = (url or "").lower().strip()
    doc_len = doc_lengths.get(doc_key, len(doc_tokens))
    avg_doc_len = sum(doc_lengths.values()) / max(len(doc_lengths), 1)
    if avg_doc_len <= 0:
        avg_doc_len = 1.0

    score = 0.0
    for term in query_terms:
        tf = tf_counts.get(term, 0)
        if tf <= 0:
            continue
        df = doc_freqs.get(term, 0)
        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        denom = tf + k1 * (1.0 - b + b * (doc_len / avg_doc_len))
        score += idf * ((tf * (k1 + 1.0)) / max(denom, 1e-9))
    return score


def filter_by_bm25(
    results: list[SearchResult],
    original_query: str,
    threshold: float = 1.2,
) -> list[SearchResult]:
    if not results:
        print("BM25 score distribution: no results")
        return []

    n_docs = len(results)
    doc_lengths: dict[str, int] = {}
    doc_freqs: dict[str, int] = defaultdict(int)

    for result in results:
        doc_key = (result.url or "").lower().strip()
        doc_tokens = tokenize(f"{result.title or ''} {result.snippet or ''}")
        doc_lengths[doc_key] = len(doc_tokens)
        for term in set(doc_tokens):
            doc_freqs[term] += 1

    scored: list[tuple[SearchResult, float]] = []
    for result in results:
        score = bm25_score(
            original_query,
            result.title or "",
            result.snippet or "",
            result.url or "",
            doc_lengths,
            doc_freqs,
            n_docs,
        )
        scored.append((result, score))

    scores = [s for _, s in scored]

    # Dynamic threshold: lower bound is 30% of max score in pool
    # prevents over-filtering on short/simple queries with low absolute BM25 scores
    dynamic_threshold = min(threshold, max(scores) * 0.30)

    print(
        f"BM25 score distribution: min={min(scores):.3f}, max={max(scores):.3f}, "
        f"mean={mean(scores):.3f}, effective_threshold={dynamic_threshold:.3f}"
    )
    return [result for result, score in scored if score >= dynamic_threshold]


__all__ = ["filter_by_bm25", "tokenize"]

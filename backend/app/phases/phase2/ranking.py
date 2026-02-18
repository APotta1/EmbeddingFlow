import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

from app.phases.phase1.llm_utils import get_client, get_model, parse_llm_response
from app.phases.phase2.schemas import SearchResult

# Default: take top 20; caller can pass 15–20
DEFAULT_TOP_N = 20

# Domains with Groq credibility below this (after 0–10 → 0–1) are filtered out
CREDIBILITY_MIN = 0.2

# Max domains to send in one Groq call (avoid token limits)
GROQ_DOMAINS_BATCH = 50

# Max results per LLM call for relevance filtering (avoids token limits)
RELEVANCE_LLM_BATCH = 25

RELEVANCE_SYSTEM = """You judge whether search results are relevant to the user's query.
You will see a user query and a numbered list of search results (title + short snippet).
Return a JSON array of the 1-based indices that are RELEVANT to the query (results that clearly address or relate to what the user asked).
Exclude results that are off-topic, only share a word by coincidence, or are about a different meaning of the query (e.g. "technology integration" when the user asked about "integral calculus").
Return only the array, e.g. [1, 3, 5, 12]. No explanation or markdown."""

CREDIBILITY_SYSTEM = """You assess the credibility of web domains for search result ranking.
Given a list of domains, return a JSON object mapping each domain to a credibility score from 0 to 10:
- 10: Highly trusted (e.g. major news, .gov, .edu, established publishers, academic)
- 7–9: Generally reliable (reputable brands, known outlets)
- 4–6: Mixed or unknown (blogs, smaller sites, unclear reputation)
- 1–3: Low credibility (content farms, spam, clickbait, misleading)
- 0: Exclude (spam, unsafe, or not suitable for factual content)

Use the domain string exactly as given. Return only valid JSON: {"domain.example.com": 8, ...}. No markdown or explanation."""


def _domain_from_result(result: SearchResult) -> str:
    """Normalized domain for scoring/filtering."""
    d = (result.domain or "").strip().lower()
    if d:
        return d
    try:
        return urlparse(result.url).netloc.lower() or ""
    except Exception:
        return ""


def _get_domain_credibility_groq(domains: list[str]) -> dict[str, float]:
    """
    Call Groq to assess credibility of each domain. Returns map domain -> 0.0–1.0.
    Missing or invalid entries default to 0.5 (neutral).
    """
    if not domains:
        return {}
    # Deduplicate and limit batch size
    unique = list(dict.fromkeys(d for d in domains if d))
    if not unique:
        return {}
    batch = unique[:GROQ_DOMAINS_BATCH]

    user_content = "Rate credibility (0–10) for each of these domains:\n" + ", ".join(batch)

    try:
        client = get_client()
        response = client.chat.completions.create(
            model=get_model(),
            messages=[
                {"role": "system", "content": CREDIBILITY_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content
        raw = parse_llm_response(content or "{}")
    except Exception as e:
        print(f"Groq domain credibility error: {e}")
        return {}

    out: dict[str, float] = {}
    for domain, value in (raw or {}).items():
        if not isinstance(domain, str):
            continue
        d = domain.strip().lower()
        if not d:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        # Normalize 0–10 to 0–1
        out[d] = max(0.0, min(1.0, score / 10.0))
    return out


def _score_position(result: SearchResult, max_position: int = 50) -> float:
    """Better (lower) position in search results → higher score. 0–1 range."""
    if max_position <= 0:
        return 1.0
    p = max(1, result.position)
    return max(0.0, 1.0 - (p - 1) / max_position)


def _score_domain_authority(result: SearchResult, credibility_map: dict[str, float]) -> float:
    """Use Groq-assessed credibility; default 0.5 if domain not in map."""
    domain = _domain_from_result(result)
    if not domain:
        return 0.5
    return credibility_map.get(domain, 0.5)


def _parse_date(published_date: Optional[str]) -> Optional[datetime]:
    """Best-effort parse of published_date for recency scoring."""
    if not published_date or not published_date.strip():
        return None
    s = published_date.strip()
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    m = re.search(r"(\d{4})", s)
    if m:
        try:
            return datetime(int(m.group(1)), 1, 1)
        except ValueError:
            pass
    return None


def _score_recency(result: SearchResult, time_sensitive: bool) -> float:
    """Prefer newer content when time_sensitive; else neutral. 0–1 range."""
    if not time_sensitive:
        return 0.5
    dt = _parse_date(result.published_date)
    if not dt:
        return 0.5
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    age_days = (now - dt).days
    if age_days <= 0:
        return 1.0
    if age_days <= 30:
        return 0.95
    if age_days <= 90:
        return 0.8
    if age_days <= 365:
        return 0.5
    return 0.2


def _filter_by_credibility(
    results: list[SearchResult],
    credibility_map: dict[str, float],
    min_credibility: float = CREDIBILITY_MIN,
) -> list[SearchResult]:
    """Drop results whose domain has credibility below threshold. Unknown domains kept."""
    out = []
    for r in results:
        domain = _domain_from_result(r)
        if not domain:
            out.append(r)
            continue
        score = credibility_map.get(domain)
        if score is None:
            out.append(r)  # no score = keep (neutral)
            continue
        if score < min_credibility:
            continue
        out.append(r)
    return out


def _basic_checks(result: SearchResult) -> bool:
    """Filter out invalid or non-HTTPS URLs. Only HTTPS links are kept."""
    if not result.url or not result.url.strip():
        return False
    url_lower = result.url.lower().strip()
    if not url_lower.startswith("https://"):
        return False
    if len(url_lower) > 2048:
        return False
    return True


def _filter_by_llm_relevance(results: list[SearchResult], original_query: str) -> list[SearchResult]:
    """
    Use Groq to keep only results that are relevant to the query. Batches results to stay under context.
    Returns the subset of results the LLM marked as relevant.
    """
    if not results or not original_query or not original_query.strip():
        return results

    out: list[SearchResult] = []
    snippet_max = 250  # chars per snippet to save tokens

    for start in range(0, len(results), RELEVANCE_LLM_BATCH):
        batch = results[start : start + RELEVANCE_LLM_BATCH]
        lines = [f"{i+1}. Title: {(r.title or '')[:120]}\n   Snippet: {(r.snippet or '')[:snippet_max]}" for i, r in enumerate(batch)]
        user_content = f"User query: {original_query}\n\nSearch results:\n" + "\n\n".join(lines) + "\n\nReturn a JSON array of the 1-based indices that are relevant to the query (e.g. [1, 2, 5])."

        try:
            client = get_client()
            response = client.chat.completions.create(
                model=get_model(),
                messages=[
                    {"role": "system", "content": RELEVANCE_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.1,
            )
            content = response.choices[0].message.content
            raw = parse_llm_response(content or "[]")
        except Exception as e:
            print(f"Groq relevance filter error: {e}")
            out.extend(batch)
            continue

        if isinstance(raw, list):
            indices_1based = {int(x) for x in raw if isinstance(x, (int, float))}
        elif isinstance(raw, dict):
            indices_1based = set()
            for k, v in raw.items():
                try:
                    if v and (v is True or (isinstance(v, (int, float)) and v > 0)):
                        indices_1based.add(int(k) if isinstance(k, (int, float)) else int(k.strip()))
                except (ValueError, TypeError):
                    pass
        else:
            out.extend(batch)
            continue

        for i, r in enumerate(batch):
            if (i + 1) in indices_1based:
                out.append(r)
    return out


def _score_relevance_to_query(result: SearchResult, original_query: str) -> float:
    """Score 0–1 by how many query terms appear in title/snippet; title matches count more."""
    if not original_query or not original_query.strip():
        return 0.5
    words = [w for w in re.findall(r"\w+", original_query.lower()) if len(w) > 1]
    if not words:
        return 0.5
    title_lower = (result.title or "").lower()
    snippet_lower = (result.snippet or "").lower()
    score = 0.0
    for w in words:
        if w in title_lower:
            score += 1.5
        elif w in snippet_lower:
            score += 1.0
    return min(1.0, score / len(words))


WEIGHT_POSITION = 0.30
WEIGHT_DOMAIN = 0.35
WEIGHT_RECENCY = 0.15
WEIGHT_RELEVANCE = 0.20

# Drop results with relevance to original query below this (removes off-topic hits from any API)
MIN_RELEVANCE_TO_ORIGINAL_QUERY = 0.2


def rank_and_select(
    results: list[SearchResult],
    top_n: int = DEFAULT_TOP_N,
    time_sensitive: bool = False,
    original_query: str = "",
    *,
    weight_position: float = WEIGHT_POSITION,
    weight_domain: float = WEIGHT_DOMAIN,
    weight_recency: float = WEIGHT_RECENCY,
    weight_relevance: float = WEIGHT_RELEVANCE,
    min_credibility: float = CREDIBILITY_MIN,
    min_relevance: float = MIN_RELEVANCE_TO_ORIGINAL_QUERY,
    min_results_per_source: int = 0,
) -> list[SearchResult]:
    """
    Score each URL (position, domain credibility, recency, relevance to query),
    filter non-HTTPS, low-credibility, and off-topic; sort by combined score; return top N.

    Args:
        results: Merged, deduplicated SearchResult list from search.
        top_n: Number of results to return (default 20).
        time_sensitive: If True, recency scoring is applied.
        original_query: User query; used for relevance-to-query scoring.
        weight_position: Weight for position score.
        weight_domain: Weight for domain credibility score.
        weight_recency: Weight for recency score.
        weight_relevance: Weight for relevance-to-query score.
        min_credibility: Domains with Groq credibility below this (0–1) are filtered out.

    Returns:
        Top N SearchResult (HTTPS only), sorted by combined score, ready for Phase 3.
    """
    if not results:
        return []

    # Only HTTPS links
    filtered = [r for r in results if _basic_checks(r)]
    if not filtered:
        return []

    query = original_query or ""

    # LLM-based relevance filter: keep only results the model says are relevant to the query
    if query:
        filtered = _filter_by_llm_relevance(filtered, query)
    if not filtered:
        return []

    # Optional: also drop results with very low regex-based relevance score
    if query and min_relevance > 0:
        filtered = [r for r in filtered if _score_relevance_to_query(r, query) >= min_relevance]
    if not filtered:
        return []

    domains = [_domain_from_result(r) for r in filtered]
    credibility_map = _get_domain_credibility_groq(domains)
    filtered = _filter_by_credibility(filtered, credibility_map, min_credibility=min_credibility)

    if not filtered:
        return []

    total_w = weight_position + weight_domain + weight_recency + weight_relevance
    if total_w <= 0:
        total_w = 1.0
    wp = weight_position / total_w
    wd = weight_domain / total_w
    wr = weight_recency / total_w
    wrel = weight_relevance / total_w
    max_pos = max(r.position for r in filtered)

    def combined_score(r: SearchResult) -> float:
        pos_s = _score_position(r, max_position=max_pos)
        dom_s = _score_domain_authority(r, credibility_map)
        rec_s = _score_recency(r, time_sensitive)
        rel_s = _score_relevance_to_query(r, query)
        return wp * pos_s + wd * dom_s + wr * rec_s + wrel * rel_s

    filtered.sort(key=combined_score, reverse=True)
    top = filtered[:top_n]

    # Optional: ensure at least min_results_per_source from each API (so one doesn't dominate)
    if min_results_per_source > 0 and len(top) == top_n:
        top_set = set(id(r) for r in top)
        by_source: dict[str, list[SearchResult]] = {}
        for r in filtered:
            key = (r.source_api or "serper").lower()
            by_source.setdefault(key, []).append(r)
        for source, results_from_source in by_source.items():
            in_top = [r for r in top if (r.source_api or "").lower() == source]
            if len(in_top) >= min_results_per_source or not results_from_source:
                continue
            needed = min_results_per_source - len(in_top)
            # Best from this source not already in top
            candidates = [r for r in results_from_source if id(r) not in top_set][:needed]
            if not candidates:
                continue
            # Remove lowest-scoring from top to make room (so we keep top_n)
            top_sorted = sorted(top, key=combined_score)
            to_remove = [r for r in top_sorted if id(r) not in {id(c) for c in candidates}][:needed]
            for r in to_remove:
                top.remove(r)
                top_set.discard(id(r))
            for r in candidates:
                top.append(r)
                top_set.add(id(r))
        top.sort(key=combined_score, reverse=True)
        top = top[:top_n]

    for i, r in enumerate(top, start=1):
        r.position = i

    return top

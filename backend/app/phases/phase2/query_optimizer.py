from dataclasses import dataclass

from app.phases.phase1.schemas import Phase2Payload


def normalize_query(query: str) -> str:
    return query.lower().strip()


def _is_duplicate_query(query: str, seen_normalized: set[str]) -> bool:
    """True only on exact normalized match. Substring heuristics wrongly collapse distinct sub-questions."""
    return normalize_query(query) in seen_normalized


def _diversity_score(candidate: str, seen_normalized: set[str]) -> float:
    """Higher = more different from what we already have. Prefer higher."""
    c = normalize_query(candidate)
    if not c or c in seen_normalized:
        return -1.0
    words = set(c.split())
    if not words:
        return 0.0
    min_overlap = 1.0
    for s in seen_normalized:
        sw = set(s.split())
        overlap = len(words & sw) / len(words) if words else 0
        min_overlap = min(min_overlap, overlap)
    return 1.0 - min_overlap


@dataclass
class OptimizedQueriesResult:
    queries: list[str]
    duplicate_to_canonical: dict[str, str]


def _record_duplicate(
    duplicate_to_canonical: dict[str, str],
    canonical_by_norm: dict[str, str],
    n: str,
) -> None:
    """Map normalized query to the first query string that was kept for that norm."""
    if n not in duplicate_to_canonical and n in canonical_by_norm:
        duplicate_to_canonical[n] = canonical_by_norm[n]


def optimize_queries(payload: Phase2Payload) -> OptimizedQueriesResult:
    """
    When Phase 1 set ``tavily_queries`` / ``serper_queries``, return those only (deduped,
    Tavily order then Serper)—do not prepend ``original_query`` or merge legacy fields.
    """
    tav = [q.strip() for q in (payload.tavily_queries or []) if q and q.strip()]
    ser = [q.strip() for q in (payload.serper_queries or []) if q and q.strip()]
    if tav or ser:
        queries: list[str] = []
        seen: set[str] = set()
        duplicate_to_canonical: dict[str, str] = {}
        canonical_by_norm: dict[str, str] = {}
        for q in tav + ser:
            n = normalize_query(q)
            if n in seen:
                _record_duplicate(duplicate_to_canonical, canonical_by_norm, n)
                continue
            queries.append(q)
            seen.add(n)
            canonical_by_norm[n] = q
        return OptimizedQueriesResult(queries=queries, duplicate_to_canonical=duplicate_to_canonical)

    queries: list[str] = []
    seen: set[str] = set()
    duplicate_to_canonical: dict[str, str] = {}
    canonical_by_norm: dict[str, str] = {}

    if payload.original_query and payload.original_query.strip():
        q = payload.original_query.strip()
        n = normalize_query(q)
        queries.append(q)
        seen.add(n)
        canonical_by_norm[n] = q

    for sq in payload.subqueries[:3]:
        if not sq or not sq.strip():
            continue
        q = sq.strip()
        n = normalize_query(q)
        if _is_duplicate_query(q, seen):
            _record_duplicate(duplicate_to_canonical, canonical_by_norm, n)
            continue
        queries.append(q)
        seen.add(n)
        canonical_by_norm[n] = q

    MAX_VARIANTS = 3
    scored: list[tuple[float, str]] = []
    for sv in payload.search_variants:
        sv = (sv or "").strip()
        if not sv:
            continue
        n = normalize_query(sv)
        if n in seen:
            _record_duplicate(duplicate_to_canonical, canonical_by_norm, n)
            continue
        score = _diversity_score(sv, seen)
        if score >= 0:
            scored.append((score, sv))

    scored.sort(key=lambda x: -x[0])
    for score, sv in scored[:MAX_VARIANTS]:
        if len(queries) >= 7:
            break
        n = normalize_query(sv)
        queries.append(sv)
        seen.add(n)
        canonical_by_norm[n] = sv

    return OptimizedQueriesResult(queries=queries, duplicate_to_canonical=duplicate_to_canonical)

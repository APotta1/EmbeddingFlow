from dataclasses import dataclass
from app.phases.phase1.schemas import Phase2Payload


def normalize_query(query: str) -> str:
    return query.lower().strip()


def _is_duplicate_query(query: str, seen_normalized: set[str]) -> bool:
    n = normalize_query(query)
    if n in seen_normalized:
        return True
    for existing in seen_normalized:
        if len(existing) > 10 and (n in existing or existing in n):
            return True
    return False


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


def optimize_queries(payload: Phase2Payload) -> OptimizedQueriesResult:
    queries: list[str] = []
    seen: set[str] = set[str]()
    duplicate_to_canonical: dict[str, str] = {}

    # Original query
    if payload.original_query and payload.original_query.strip():
        q = payload.original_query.strip()
        queries.append(q)
        seen.add(normalize_query(q))

    # Subqueries
    for sq in payload.subqueries[:3]:
        if not sq or not sq.strip():
            continue
        q = sq.strip()
        n = normalize_query(q)
        if _is_duplicate_query(q, seen):
            if n not in duplicate_to_canonical and queries:
                duplicate_to_canonical[n] = queries[0]
            continue
        queries.append(q)
        seen.add(n)


    MAX_VARIANTS = 3
    scored: list[tuple[float, str]] = []
    for sv in payload.search_variants:
        sv = (sv or "").strip()
        if not sv:
            continue
        n = normalize_query(sv)
        if n in seen:
            if n not in duplicate_to_canonical and queries:
                duplicate_to_canonical[n] = queries[0]
            continue
        score = _diversity_score(sv, seen)
        if score >= 0:
            scored.append((score, sv))

    scored.sort(key=lambda x: -x[0])
    for score, sv in scored[:MAX_VARIANTS]:
        if len(queries) >= 7:
            break
        queries.append(sv)
        seen.add(normalize_query(sv))

    return OptimizedQueriesResult(queries=queries, duplicate_to_canonical=duplicate_to_canonical) 
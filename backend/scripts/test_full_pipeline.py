"""
Test full pipeline: Phase 1 (analyze → decompose → expand) → Phase 2 (search → rank).

Run from backend with:
  python scripts/test_full_pipeline.py
  python scripts/test_full_pipeline.py "Your search query here"

Requires: GROQ_API_KEY, TAV_API_KEY, SERP_API_KEY in env (or .env).
Prints nitty-gritty: Phase 1 output, optimized queries, search results by source,
then ranked top N with domain/source for tuning.
"""

import os
import sys
from textwrap import shorten

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add backend root so "app" is importable from scripts/ or from backend/
_backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

from app.phases.phase1.pipeline import run_phase1, to_phase2_payload
from app.phases.phase2.query_optimizer import optimize_queries
from app.phases.phase2.ranking import rank_and_select
from app.phases.phase2.search_orchestrator import search_parallel


def _trunc(s: str, max_len: int = 72) -> str:
    return shorten(s, width=max_len, placeholder="…") if s else ""


def _section(title: str) -> None:
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def _sub(title: str) -> None:
    print(f"\n--- {title} ---")


def main() -> None:
    query = (sys.argv[1] if len(sys.argv) > 1 else "How does the Federal Reserve control inflation?").strip()
    if not query:
        query = "How does the Federal Reserve control inflation?"

    # Env check
    missing = [k for k in ("GROQ_API_KEY", "TAV_API_KEY", "SERP_API_KEY") if not os.environ.get(k)]
    if missing:
        print("Missing env vars (set or use .env):", ", ".join(missing))
        sys.exit(1)

    _section("PHASE 1: Query processing")
    print(f"Input query: {query}")

    phase1 = run_phase1(query)
    payload = to_phase2_payload(query, phase1)

    _sub("Intent & entities")
    print(f"  Intent: {phase1.query_analysis.intent.primary}")
    print(f"  Time sensitive: {phase1.query_analysis.time_sensitive}")
    print(f"  Entities: {[e.text for e in phase1.query_analysis.entities]}")
    print(f"  Complexity: {phase1.query_analysis.complexity.level}")

    _sub("Subqueries (decomposition)")
    for i, sq in enumerate(phase1.query_decomposition.sub_questions, 1):
        print(f"  {i}. {_trunc(sq, 70)}")

    _sub("Search variants (expansion)")
    for i, v in enumerate(phase1.query_expansion.search_variations, 1):
        print(f"  {i}. {_trunc(v, 70)}")

    _sub("Phase 2 payload (summary)")
    print(f"  original_query: {_trunc(payload.original_query)}")
    print(f"  intent: {payload.intent}")
    print(f"  subqueries count: {len(payload.subqueries)}")
    print(f"  search_variants count: {len(payload.search_variants)}")
    print(f"  time_sensitive: {payload.time_sensitivity.is_time_sensitive}")
    print(f"  max_results_per_query: {payload.constraints.max_results_per_query}")

    _section("PHASE 2: Query optimization")
    optimized = optimize_queries(payload)
    print(f"Optimized queries ({len(optimized.queries)}):")
    for i, q in enumerate(optimized.queries, 1):
        print(f"  {i}. {_trunc(q, 70)}")
    if optimized.duplicate_to_canonical:
        print("Duplicate → canonical:", optimized.duplicate_to_canonical)

    _section("PHASE 2: Search (Tavily + Serper, parallel)")
    search_output = search_parallel(payload)
    print(f"Queries used: {search_output.queries_used}")
    print(f"Total search tasks (queries × APIs): {search_output.total_searched}")
    print(f"Merged + deduped URL count: {len(search_output.urls)}")

    _sub("All merged results (before ranking) — source & domain")
    print(f"{'#':>3}  {'source':<8}  {'domain':<32}  title")
    print("-" * 80)
    for i, r in enumerate(search_output.urls, 1):
        dom = (r.domain or "")[:30] if r.domain else "(no domain)"
        print(f"{i:>3}  {r.source_api:<8}  {dom:<32}  {_trunc(r.title or '', 36)}")

    _sub("Source breakdown")
    by_api = {}
    for r in search_output.urls:
        by_api[r.source_api] = by_api.get(r.source_api, 0) + 1
    for api, count in sorted(by_api.items()):
        print(f"  {api}: {count} URLs")

    _section("PHASE 2: Ranking (Groq credibility + position + recency)")
    ranked = rank_and_select(
        search_output.urls,
        top_n=20,
        time_sensitive=payload.time_sensitivity.is_time_sensitive,
        original_query=payload.original_query,
    )
    print(f"After ranking: {len(ranked)} URLs (top_n=20)")
    print(f"Time sensitive used for recency: {payload.time_sensitivity.is_time_sensitive}")

    _sub("Final ranked URLs (for Phase 3 extraction)")
    print(f"{'#':>3}  {'source':<8}  {'domain':<28}  title")
    print("-" * 80)
    for r in ranked:
        dom = (r.domain or "")[:26] if r.domain else "(no domain)"
        print(f"{r.position:>3}  {r.source_api:<8}  {dom:<28}  {_trunc(r.title or '', 32)}")

    _sub("Final URLs (raw)")
    for r in ranked:
        print(f"  {r.url}")

    _section("DONE")
    print(f"Query: {query}")
    print(f"Phase 1 → Phase 2 payload → {len(search_output.urls)} merged → {len(ranked)} ranked")
    print()


if __name__ == "__main__":
    main()

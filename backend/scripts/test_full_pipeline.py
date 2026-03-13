"""
Test full pipeline:
- Phase 1: analyze → decompose → expand
- Phase 2: search → rank
- Phase 3: fetch → extract → clean
- Phase 4: chunk → contextualize

Run from backend with:
  python scripts/test_full_pipeline.py
  python scripts/test_full_pipeline.py "Your search query here"

Requires: GROQ_API_KEY, TAV_API_KEY, SERP_API_KEY in env (or .env).
Prints nitty-gritty: Phase 1 output, optimized queries, search results by source,
then ranked top N with domain/source, Phase 3 extraction stats/doc summaries,
and Phase 4 chunking/contextualization summaries.
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
from app.phases.phase2.schemas import Phase2Output
from app.phases.phase2.search_orchestrator import search_parallel
from app.phases.phase3.pipeline import run_phase3
from app.phases.phase4.pipeline import run_phase4  # type: ignore[import-not-found]
from app.phases.phase4.schemas import Phase4Config  # type: ignore[import-not-found]


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
        min_results_per_source=5,
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

    _section("PHASE 3: Content extraction & cleaning")
    phase2_for_phase3 = Phase2Output(
        original_query=payload.original_query,
        urls=ranked,
        total_searched=search_output.total_searched,
        queries_used=search_output.queries_used,
    )
    phase3 = run_phase3(phase2_for_phase3)

    _sub("Phase 3 stats")
    stats = phase3.stats
    print(f"  total_input_urls: {stats.total_input_urls}")
    print(f"  fetched: {stats.fetched}")
    print(f"  successful: {stats.successful}")
    print(f"  skipped_robots: {stats.skipped_robots}")
    print(f"  skipped_nontext: {stats.skipped_nontext}")
    print(f"  failed: {stats.failed}")
    print(f"  below_quality_threshold: {stats.below_quality_threshold}")

    _sub("Extracted documents (summary)")
    print(f"{'#':>3}  {'domain':<28}  {'source':<8}  {'words':>6}  title")
    print("-" * 80)
    for i, doc in enumerate(phase3.documents, 1):
        dom = (doc.domain or "")[:26] if doc.domain else "(no domain)"
        src = (doc.source_api or "")[:6] if doc.source_api else ""
        word_count = len(doc.content.split())
        print(f"{i:>3}  {dom:<28}  {src:<8}  {word_count:>6}  {_trunc(doc.title or '', 32)}")

    _section("PHASE 4: Contextual chunking")
    # Use default Phase4Config; adjust here if you want to experiment.
    phase4_config = Phase4Config()
    phase4 = run_phase4(phase3, config=phase4_config)

    _sub("Phase 4 stats")
    cstats = phase4.stats
    print(f"  total_documents: {cstats.total_documents}")
    print(f"  total_chunks: {cstats.total_chunks}")
    print(f"  contextualized_chunks: {cstats.contextualized_chunks}")
    print(f"  failed_contextualizations: {cstats.failed_contextualizations}")

    _sub("Sample chunks (raw vs contextualized)")
    sample_chunks = phase4.chunks[:5]
    for i, ch in enumerate(sample_chunks, 1):
        print(f"\nChunk {i} (doc_index={ch.document_index}, chunk_index={ch.chunk_index})")
        print(f"  URL: {ch.url}")
        print(f"  approx_token_count: {ch.approx_token_count}")
        print("  Raw text:")
        print(f"    {_trunc(ch.raw_text, 160)}")
        if ch.contextualized_text != ch.raw_text:
            print("  Contextualized text:")
            print(f"    {_trunc(ch.contextualized_text, 200)}")

    _section("DONE")
    print(f"Query: {query}")
    print(
        f"Phase 1 → Phase 2 payload → {len(search_output.urls)} merged → "
        f"{len(ranked)} ranked → {len(phase3.documents)} extracted → "
        f"{len(phase4.chunks)} chunks"
    )
    print()


if __name__ == "__main__":
    main()

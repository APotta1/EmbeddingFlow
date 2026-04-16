"""
Test pipeline phases 1–4 (Phase 5 embedding/indexing is not run here).

- Phase 1: parallel analyze + HyDE/retrieval plan → Phase 2 payload
- Phase 2: search → rank
- Phase 3: fetch → extract → clean
- Phase 4: chunk → contextualize

Run from backend with:
  python scripts/test_full_pipeline.py
  python scripts/test_full_pipeline.py "Your search query here"

Requires: GROQ_API_KEY, TAV_API_KEY, SERP_API_KEY in env (or .env).
Prints Phase 1 output, optimized queries, search results, ranked URLs,
Phase 3 extraction stats, and Phase 4 chunking/contextualization summaries.
"""

import os
import sys
from collections import Counter
from textwrap import shorten
from urllib.parse import urlparse

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
from app.phases.phase2 import ranking as ranking_mod
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


def _result_domain(r) -> str:
    if r.domain and str(r.domain).strip():
        return str(r.domain).strip().lower()
    try:
        return urlparse(r.url or "").netloc.lower() or (r.url or "")
    except Exception:
        return (r.url or "").lower()


def main() -> None:
    query = (sys.argv[1] if len(sys.argv) > 1 else "How does the Federal Reserve control inflation?").strip()
    if not query:
        query = "How does the Federal Reserve control inflation?"

    # Env check
    missing = [
        k
        for k in ("GROQ_API_KEY", "TAV_API_KEY", "SERP_API_KEY")
        if not os.environ.get(k)
    ]
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

    _sub("HyDE + retrieval plan")
    plan = phase1.query_retrieval_plan
    print(f"  HyDE (truncated): {_trunc(plan.hyde_document, 120)}")
    for i, k in enumerate(plan.keyword_variants, 1):
        print(f"  Keyword {i}: {_trunc(k, 70)}")
    for i, sq in enumerate(plan.sub_questions, 1):
        print(f"  Sub-question {i}: {_trunc(sq, 70)}")

    _sub("Phase 2 payload (summary)")
    print(f"  original_query: {_trunc(payload.original_query)}")
    print(f"  intent: {payload.intent}")
    print(f"  subqueries count: {len(payload.subqueries)}")
    print(f"  search_variants count: {len(payload.search_variants)}")
    print(f"  tavily_queries: {len(payload.tavily_queries)}  serper_queries: {len(payload.serper_queries)}")
    print(f"  time_sensitive: {payload.time_sensitivity.is_time_sensitive}")
    print(f"  max_results_per_query: {payload.constraints.max_results_per_query}")

    _sub("Phase 1 validation")
    level = phase1.query_analysis.complexity.level.lower().strip()
    time_s = phase1.query_analysis.time_sensitive
    oq_norm = payload.original_query.strip()

    if level == "simple":
        assert len(payload.subqueries) == 0, (
            f"FAIL: simple query should have 0 subqueries, got {len(payload.subqueries)}"
        )
        n_ser = len(payload.serper_queries)
        assert n_ser <= 1, f"FAIL: simple should have at most 1 Serper query, got {n_ser}"
        assert len(payload.tavily_queries) >= 1, "FAIL: simple should have at least 1 Tavily query (HyDE)"
        if time_s:
            assert n_ser == 1, (
                f"FAIL: simple+time_sensitive should send 1 Serper query for freshness, got {n_ser}"
            )
        else:
            if plan.keyword_variants:
                assert n_ser == 1, (
                    f"FAIL: simple with keyword variants should send 1 Serper, got {n_ser}"
                )
            else:
                assert n_ser == 0, (
                    f"FAIL: simple without keyword variants should have 0 Serper, got {n_ser}"
                )
        print("  [PASS] simple: 0 subqueries, Tavily+Serper counts match policy")

    elif level == "moderate":
        assert len(payload.subqueries) == 0, (
            f"FAIL: moderate should have 0 subqueries, got {len(payload.subqueries)}"
        )
        assert len(payload.tavily_queries) == 1, (
            f"FAIL: moderate should have 1 Tavily query (HyDE), got {len(payload.tavily_queries)}"
        )
        assert len(payload.serper_queries) <= 3, (
            f"FAIL: moderate should have <=3 Serper queries, got {len(payload.serper_queries)}"
        )
        print(
            f"  [PASS] moderate: 0 subqueries, {len(payload.tavily_queries)} Tavily, "
            f"{len(payload.serper_queries)} Serper"
        )

    elif level == "complex":
        assert len(payload.subqueries) > 0, "FAIL: complex should have subqueries"
        assert len(payload.subqueries) <= 4, (
            f"FAIL: complex should have <=4 subqueries, got {len(payload.subqueries)}"
        )
        print(
            f"  [PASS] complex: {len(payload.subqueries)} subqueries, "
            f"{len(payload.tavily_queries)} Tavily, {len(payload.serper_queries)} Serper"
        )

    for sq in payload.subqueries:
        assert sq.strip() != oq_norm, (
            f"FAIL: original_query leaked into subqueries: {sq!r}"
        )
    print("  [PASS] no original_query padding in subqueries")

    assert payload.hyde_document and len(payload.hyde_document) > 50, (
        "FAIL: HyDE document too short or empty"
    )
    print(f"  [PASS] HyDE document present ({len(payload.hyde_document)} chars)")

    _section("PHASE 2: Query optimization")
    optimized = optimize_queries(payload)
    print(f"Optimized queries ({len(optimized.queries)}):")
    for i, q in enumerate(optimized.queries, 1):
        print(f"  {i}. {_trunc(q, 70)}")
    if optimized.duplicate_to_canonical:
        print(f"  ({len(optimized.duplicate_to_canonical)} exact duplicate(s) removed)")
        for dup_norm, canonical in optimized.duplicate_to_canonical.items():
            print(
                f"    REMOVED norm: '{_trunc(dup_norm, 50)}' "
                f"→ kept: '{_trunc(canonical, 50)}'"
            )

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

    _sub("BM25 score distribution (from merge step)")
    print("  (distribution printed inline above during _merge_results)")
    print(f"  URLs surviving BM25 threshold: {len(search_output.urls)}")
    if len(search_output.urls) == 0:
        print("  [WARN] BM25 filtered everything — threshold may be too aggressive")
    elif len(search_output.urls) < 5:
        print(f"  [WARN] only {len(search_output.urls)} URLs survived BM25 — consider lowering threshold")
    else:
        print(f"  [PASS] {len(search_output.urls)} URLs passed BM25 threshold")

    _section("PHASE 2: Ranking (Groq credibility + position + recency)")
    ranked = rank_and_select(
        search_output.urls,
        top_n=20,
        time_sensitive=payload.time_sensitivity.is_time_sensitive,
        original_query=payload.original_query,
        hyde_document=payload.hyde_document,
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

    _sub("Ranking validation")
    domains_in_ranked = [_result_domain(r) for r in ranked]

    blocked_found = [
        d for d in domains_in_ranked if d and ranking_mod._domain_is_low_value_for_research(d)
    ]
    if blocked_found:
        uniq = sorted(set(blocked_found))
        print(f"  [FAIL] low-value / blocked-style domains in ranked results: {uniq[:12]}")
    else:
        print("  [PASS] no ranking blocklist domains in ranked results")

    tavily_count = sum(1 for r in ranked if r.source_api == "tavily")
    serper_count = sum(1 for r in ranked if r.source_api == "serper")
    print(f"  source mix: tavily={tavily_count}, serper={serper_count}")
    if serper_count == 0 and level != "simple":
        print(f"  [WARN] Serper contributed 0 URLs to final ranking for {level} query")

    top5_domains = domains_in_ranked[:5]
    top5_counts = Counter(top5_domains)
    warned_dom = False
    for domain, count in top5_counts.items():
        if count > 2:
            print(f"  [WARN] {domain!r} appears {count} times in top 5")
            warned_dom = True
    if not warned_dom and top5_domains:
        print("  [PASS] domain diversity ok in top 5 (no domain >2 of 5)")

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



if __name__ == "__main__":
    main()

"""
Phase 1: HyDE + keyword variants + optional multi-hop sub-questions in a single LLM call.

Runs in parallel with query_analysis (see pipeline.py). Output is clamped post-hoc using
query_analysis.complexity so routing matches the authoritative complexity level.
"""

from __future__ import annotations

from .llm_utils import get_client, get_model, parse_llm_response
from .schemas import QueryRetrievalPlanOutput

RETRIEVAL_STRATEGY_SYSTEM = """You plan web retrieval for a search pipeline using HyDE (Hypothetical Document Embeddings).

Always return valid JSON with exactly these keys:
1. hyde_document: one coherent hypothetical passage (2–6 sentences) written as if it were an expert answer or encyclopedia excerpt that directly addresses the user's query. Factual tone; no meta commentary ("Here is...", "The user asks..."). This text will be used as the semantic search string for deep retrieval (it should read like real web documents, not like a question).

2. keyword_variants: array of 1–3 SHORT keyword-style search queries (2–8 words each): entity-heavy, no full sentences, optimized for traditional web search (e.g. reference pages, canonical docs). Always include at least 1 variant (e.g. "topic definition", "topic overview") even for definitional queries—Serper indexes differently than HyDE-on-Tavily. Each keyword_variant must directly address the user's specific question and core entities; do not drift to adjacent subtopics, tangents, or broader themes the user did not ask about.

3. sub_questions: array of 0–4 distinct, self-contained questions that break multi-hop dependencies (e.g. "Who is the CEO of X?" and "What has X said about Y?"). Use [] for single-fact or purely definitional queries. Never duplicate HyDE as a question.

4. temporal_constraints: if the query is time-sensitive (news, recent, specific dates), 1–5 strings like "2024", "last month"; otherwise [].

No markdown or explanation. JSON only."""

RETRIEVAL_STRATEGY_USER = """User query:
{query}

Produce hyde_document, keyword_variants (1–3), sub_questions (0–4), temporal_constraints."""


def plan_retrieval_strategy(query: str) -> QueryRetrievalPlanOutput:
    """
    Single Groq/Llama call: HyDE passage + keyword variants + optional sub-questions + extras.
    """
    query = (query or "").strip()
    if not query:
        return QueryRetrievalPlanOutput()

    client = get_client()
    response = client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": RETRIEVAL_STRATEGY_SYSTEM},
            {"role": "user", "content": RETRIEVAL_STRATEGY_USER.format(query=query)},
        ],
        temperature=0.35,
    )
    content = response.choices[0].message.content
    raw = parse_llm_response(content)

    hyde = (raw.get("hyde_document") or "").strip()
    if not hyde:
        hyde = query

    kws = raw.get("keyword_variants") or []
    if not isinstance(kws, list):
        kws = []
    kws = [str(x).strip() for x in kws if str(x).strip()][:4]
    if not kws and query:
        # Model omitted keywords; keep Serper path viable
        kws = [" ".join(query.split()[:8]).strip() or query[:120]]

    subs = raw.get("sub_questions") or []
    if not isinstance(subs, list):
        subs = []
    subs = [str(x).strip() for x in subs if str(x).strip()][:5]

    temp = raw.get("temporal_constraints") or []
    if not isinstance(temp, list):
        temp = []
    temp = [str(x).strip() for x in temp if str(x).strip()][:8]

    return QueryRetrievalPlanOutput(
        hyde_document=hyde,
        keyword_variants=kws,
        sub_questions=subs,
        temporal_constraints=temp,
    )


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def clamp_retrieval_plan(
    original_query: str,
    complexity_level: str,
    plan: QueryRetrievalPlanOutput,
) -> QueryRetrievalPlanOutput:
    """
    Trim strategy output using Task 1.1 complexity (authoritative).

    - simple: 1 keyword variant for Serper; no sub-questions
    - moderate: HyDE + up to 3 keyword variants (no sub-questions)
    - complex: HyDE + up to 3 keyword variants + up to 4 sub-questions
    """
    original_query = (original_query or "").strip()
    level = (complexity_level or "simple").lower()
    hyde = (plan.hyde_document or "").strip() or original_query

    # dedupe while preserving order
    seen_k: set[str] = set()
    deduped_kws: list[str] = []
    for k in plan.keyword_variants:
        k = k.strip()
        if not k:
            continue
        n = _norm(k)
        if n in seen_k:
            continue
        seen_k.add(n)
        deduped_kws.append(k)

    seen_s: set[str] = set()
    deduped_subs: list[str] = []
    for s in plan.sub_questions:
        s = s.strip()
        if not s:
            continue
        n = _norm(s)
        if n in seen_s or n == _norm(hyde):
            continue
        seen_s.add(n)
        deduped_subs.append(s)

    if level == "simple":
        kws_simple = deduped_kws[:1]
        return QueryRetrievalPlanOutput(
            hyde_document=hyde,
            keyword_variants=kws_simple,
            sub_questions=[],
            temporal_constraints=plan.temporal_constraints,
        )

    if level == "moderate":
        kws_out = deduped_kws[:3]
        return QueryRetrievalPlanOutput(
            hyde_document=hyde,
            keyword_variants=kws_out,
            sub_questions=[],
            temporal_constraints=plan.temporal_constraints,
        )

    # complex
    subs_out = deduped_subs[:4]
    kws_out = deduped_kws[:3]
    return QueryRetrievalPlanOutput(
        hyde_document=hyde,
        keyword_variants=kws_out,
        sub_questions=subs_out[:4],
        temporal_constraints=plan.temporal_constraints,
    )


def engine_queries_for_complexity(
    level: str,
    plan: QueryRetrievalPlanOutput,
    original_query: str,
    time_sensitive: bool = False,
) -> tuple[list[str], list[str]]:
    """
    Build Tavily vs Serper query lists from clamped plan + complexity + time sensitivity.

    simple: Tavily = [HyDE]. Serper = first keyword variant if any; if time-sensitive and none,
    one compact line from ``original_query`` for freshness.
    moderate: Tavily = [HyDE]; Serper = keyword variants only.
    complex: union of HyDE + keywords + sub_questions on BOTH engines.
    """
    level = (level or "simple").lower().strip()
    hyde = (plan.hyde_document or "").strip() or (original_query or "").strip()
    kws = plan.keyword_variants
    subs = plan.sub_questions

    def dedupe_seq(items: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for x in items:
            x = (x or "").strip()
            if not x:
                continue
            n = _norm(x)
            if n in seen:
                continue
            seen.add(n)
            out.append(x)
        return out

    if level == "simple":
        serper: list[str] = []
        if kws:
            serper = dedupe_seq(kws[:1])
        elif time_sensitive and (original_query or "").strip():
            serper = dedupe_seq(
                [" ".join(original_query.split()[:8]).strip() or (original_query or "")[:120]]
            )
        return dedupe_seq([hyde]), serper

    if level == "moderate":
        return dedupe_seq([hyde]), dedupe_seq(list(kws))

    union = dedupe_seq([hyde] + list(kws) + list(subs))
    return list(union), list(union)

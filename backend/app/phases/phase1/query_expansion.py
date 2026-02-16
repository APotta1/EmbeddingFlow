"""
Phase 1, Task 1.3: Query Expansion.

- Use LLM to generate 3-5 search variations
- Add synonyms and related terms
- Include temporal constraints if time-sensitive
"""

from .llm_utils import get_client, get_model, parse_llm_response
from .schemas import QueryAnalysisOutput, QueryExpansionOutput

EXPANSION_SYSTEM = """You are a search query expander. Given a user query and its analysis, produce variations and related terms for multi-source web search.

Return only valid JSON with these keys:
1. search_variations: array of 3-5 different phrasings of the query (different word choices, question vs keyword form, broader/narrower). Each variation should be a full search query string.
2. synonyms_and_related: array of synonyms or related terms (short phrases or words) that could improve recall. Can be empty.
3. temporal_constraints: array of time-related search constraints ONLY if the query is time-sensitive (e.g. "2024", "recent", "last 12 months"). If not time-sensitive, use [].

No markdown or explanation. Valid JSON only."""

EXPANSION_USER_TEMPLATE = """Query: {query}

Analysis:
- Intent: {primary_intent}, categories: {categories}
- Time-sensitive: {time_sensitive}
- Time expressions in query: {time_expressions}
- Entities: {entity_texts}

Generate search_variations (3-5), synonyms_and_related, and temporal_constraints (only if time-sensitive)."""


def expand_query(query: str, query_analysis: QueryAnalysisOutput) -> QueryExpansionOutput:
    """
    Run Task 1.3: generate 3-5 search variations, synonyms/related terms, and optional temporal constraints.
    """
    query = (query or "").strip()
    if not query:
        return QueryExpansionOutput(
            search_variations=[query],
            synonyms_and_related=[],
            temporal_constraints=[],
        )

    intent = query_analysis.intent
    entity_texts = [e.text for e in query_analysis.entities]
    time_exprs = query_analysis.time_expressions if query_analysis.time_sensitive else []

    user_content = EXPANSION_USER_TEMPLATE.format(
        query=query,
        primary_intent=intent.primary,
        categories=", ".join(intent.categories) if intent.categories else "none",
        time_sensitive=query_analysis.time_sensitive,
        time_expressions=", ".join(time_exprs) if time_exprs else "none",
        entity_texts=", ".join(entity_texts) if entity_texts else "none",
    )

    client = get_client()
    response = client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": EXPANSION_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
    )
    content = response.choices[0].message.content
    raw = parse_llm_response(content)

    variations = raw.get("search_variations") or [query]
    if not isinstance(variations, list):
        variations = [query]
    variations = [str(v).strip() for v in variations if str(v).strip()][:10]
    if not variations:
        variations = [query]

    synonyms = raw.get("synonyms_and_related") or []
    if not isinstance(synonyms, list):
        synonyms = []
    synonyms = [str(s).strip() for s in synonyms if str(s).strip()]

    temporal = raw.get("temporal_constraints") or []
    if not query_analysis.time_sensitive:
        temporal = []
    elif not isinstance(temporal, list):
        temporal = []
    else:
        temporal = [str(t).strip() for t in temporal if str(t).strip()]

    return QueryExpansionOutput(
        search_variations=variations,
        synonyms_and_related=synonyms,
        temporal_constraints=temporal,
    )

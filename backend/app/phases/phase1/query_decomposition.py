"""
Phase 1, Task 1.2: Query Decomposition.

- Break complex queries into sub-questions
- Prioritize sub-questions (ordered list, first = highest priority)
"""

from .llm_utils import get_client, get_model, parse_llm_response
from .schemas import QueryAnalysisOutput, QueryDecompositionOutput

DECOMPOSITION_SYSTEM = """You are a query decomposer for a search pipeline. Given a user query and its intent/complexity, produce a set of sub-questions that encapsulate the topic and help the user conceptually understand what they are asking.

Rules:
- Return exactly the requested number of sub-questions (see user message). Count by complexity: simple → 3, moderate → 4, complex → 5. Keep each sub-question dense and on-topic; avoid stretching narrow topics with filler.
- ORDER BY THE USER'S MAIN ASK FIRST:
  - For how_to queries: put sub-questions that directly address steps, methods, or "how to do it" first (e.g. "What are the main solution techniques?", "What are the typical steps?"). Then add background (definition, types) if needed.
  - For factual / comparison / other: keep most foundational first (e.g. definition, then types, then examples).
- Stay focused on understanding and doing the topic. Avoid meta questions like "where to find resources" or "what tools exist" unless the query explicitly asks for learning resources, tools, or recommendations. Prefer concrete angles: main techniques, typical steps, key concepts, examples.
- Every sub-question must be self-contained and answerable by a web search. No duplicates or near-duplicates.
- Return only valid JSON: { "sub_questions": ["question 1", "question 2", ...] }. No markdown or explanation."""

DECOMPOSITION_USER_TEMPLATE = """Query: {query}

Analysis:
- Intent: {primary_intent}
- Complexity: {complexity_level} (suggested sub-questions: {suggested})
- Multi-hop: {multi_hop}

Return JSON with key "sub_questions": exactly {target_count} strings, in priority order (user's main ask first; for how_to put steps/methods before background). Focus on understanding and doing; avoid meta "resources/tools" unless the query asks for them."""


def _subquery_count_for_complexity(level: str) -> int:
    """Return 3, 4, or 5 sub-queries based on complexity (simple → 3, moderate → 4, complex → 5)."""
    return {"simple": 3, "moderate": 4, "complex": 5}.get(level.lower(), 4)


def decompose_query(query: str, query_analysis: QueryAnalysisOutput) -> QueryDecompositionOutput:
    """
    Run Task 1.2: break query into 3–5 prioritized sub-questions (by complexity) that encapsulate
    the topic, put the user's main ask first, and avoid meta "resources/tools" unless requested.
    """
    query = (query or "").strip()
    if not query:
        return QueryDecompositionOutput(sub_questions=[""])

    comp = query_analysis.complexity
    target_count = _subquery_count_for_complexity(comp.level)
    client = get_client()
    user_content = DECOMPOSITION_USER_TEMPLATE.format(
        query=query,
        primary_intent=query_analysis.intent.primary,
        complexity_level=comp.level,
        suggested=comp.suggested_sub_questions,
        multi_hop=comp.multi_hop,
        target_count=target_count,
    )
    response = client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": DECOMPOSITION_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content
    raw = parse_llm_response(content)
    sub_questions = raw.get("sub_questions") or [query]
    if not isinstance(sub_questions, list):
        sub_questions = [query]
    sub_questions = [str(q).strip() for q in sub_questions if str(q).strip()]
    if not sub_questions:
        sub_questions = [query]
    # Enforce target count: take first N, pad with original query if fewer
    sub_questions = sub_questions[:target_count]
    while len(sub_questions) < target_count:
        sub_questions.append(query)
    return QueryDecompositionOutput(sub_questions=sub_questions)

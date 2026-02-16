"""
Phase 1, Task 1.2: Query Decomposition.

- Break complex queries into sub-questions
- Prioritize sub-questions (ordered list, first = highest priority)
"""

from .llm_utils import get_client, get_model, parse_llm_response
from .schemas import QueryAnalysisOutput, QueryDecompositionOutput

DECOMPOSITION_SYSTEM = """You are a query decomposer. Given a user search query and its analysis (complexity, intent), break it into sub-questions when needed.

Rules:
- If the query is simple (single clear question), return a list with one element: the original query or a slight rewording.
- If the query is moderate or complex, break it into 2-5 focused sub-questions that together cover the user's need.
- Order sub-questions by priority: most important / foundational first, then follow-ups.
- Each sub-question must be self-contained and answerable by a search.
- Return only valid JSON: { "sub_questions": ["question 1", "question 2", ...] }. No markdown or explanation."""

DECOMPOSITION_USER_TEMPLATE = """Query: {query}

Analysis:
- Intent: {primary_intent}
- Complexity: {complexity_level} (suggested sub-questions: {suggested})
- Multi-hop: {multi_hop}

Return JSON with key "sub_questions" (array of strings, in priority order)."""


def decompose_query(query: str, query_analysis: QueryAnalysisOutput) -> QueryDecompositionOutput:
    """
    Run Task 1.2: break query into prioritized sub-questions.

    For simple queries, returns a single-item list (the query itself or a rewording).
    """
    query = (query or "").strip()
    if not query:
        return QueryDecompositionOutput(sub_questions=[""])

    comp = query_analysis.complexity
    if comp.level == "simple" and comp.suggested_sub_questions == 0:
        return QueryDecompositionOutput(sub_questions=[query])

    client = get_client()
    user_content = DECOMPOSITION_USER_TEMPLATE.format(
        query=query,
        primary_intent=query_analysis.intent.primary,
        complexity_level=comp.level,
        suggested=comp.suggested_sub_questions,
        multi_hop=comp.multi_hop,
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
    return QueryDecompositionOutput(sub_questions=sub_questions)

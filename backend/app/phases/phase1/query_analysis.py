"""
Phase 1, Task 1.1: Query Analysis.

- Intent classification (purpose of query)
- Entity extraction (important words/concepts)
- Time-sensitivity detection (time or recency)
- Complexity assessment (how difficult the query is)
"""

from .llm_utils import get_client, get_model, parse_llm_response
from .schemas import (
    ComplexityOutput,
    EntityOutput,
    IntentOutput,
    QueryAnalysisOutput,
    QueryAnalysisResponse,
)

QUERY_ANALYSIS_SYSTEM = """You are a query analyst. Analyze the user's search query and return a JSON object with:

1. intent: purpose of the query
   - primary: one of factual, comparison, how_to, opinion, news, numerical, other
   - categories: list of all applicable intents from the same set

2. entities: important words or concepts (names, products, events, places, dates, topics)
   - Each: { "text": "...", "type": "person"|"organization"|"product"|"event"|"place"|"date"|"topic"|"other" }

3. time_sensitive: true if the query asks about recent events, specific dates, or recency

4. time_expressions: list of explicit time phrases in the query if time_sensitive (e.g. "2024", "last quarter", "recent"); empty list if not time_sensitive

5. complexity:
   - level: "simple" (single fact), "moderate" (multiple aspects), "complex" (multi-part or multi-hop)
   - suggested_sub_questions: 0 for simple, 1-3 for moderate, 2-5 for complex
   - multi_hop: true if the query likely needs connecting multiple sources or steps (e.g. "What did the CEO of X say about Y?")

Return only valid JSON with keys: intent, entities, time_sensitive, time_expressions, complexity. No markdown or explanation."""

QUERY_ANALYSIS_USER_TEMPLATE = "Analyze this search query:\n\n{query}"


def analyze_query(query: str) -> QueryAnalysisResponse:
    """
    Run Task 1.1 Query Analysis on the given query string.

    Returns structured JSON-ready result for Phase 1.2/1.3 and Phase 2.
    """
    if not query or not query.strip():
        return QueryAnalysisResponse(
            query_analysis=QueryAnalysisOutput(
                intent=IntentOutput(primary="other", categories=[]),
                entities=[],
                time_sensitive=False,
                time_expressions=[],
                complexity=ComplexityOutput(level="simple", suggested_sub_questions=0, multi_hop=False),
            )
        )

    client = get_client()
    response = client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": QUERY_ANALYSIS_SYSTEM},
            {"role": "user", "content": QUERY_ANALYSIS_USER_TEMPLATE.format(query=query.strip())},
        ],
        temperature=0.1,
    )
    content = response.choices[0].message.content
    raw = parse_llm_response(content)

    # Map raw dict to Pydantic models
    intent_raw = raw.get("intent") or {}
    intent = IntentOutput(
        primary=intent_raw.get("primary", "other"),
        categories=intent_raw.get("categories", []),
    )
    entities = [
        EntityOutput(text=e["text"], type=e.get("type", "other"))
        for e in raw.get("entities", [])
    ]
    time_sensitive = raw.get("time_sensitive", False)
    time_expressions = raw.get("time_expressions", []) if time_sensitive else []
    comp = raw.get("complexity") or {}
    complexity = ComplexityOutput(
        level=comp.get("level", "simple"),
        suggested_sub_questions=comp.get("suggested_sub_questions", 0),
        multi_hop=comp.get("multi_hop", False),
    )

    return QueryAnalysisResponse(
        query_analysis=QueryAnalysisOutput(
            intent=intent,
            entities=entities,
            time_sensitive=time_sensitive,
            time_expressions=time_expressions,
            complexity=complexity,
        )
    )

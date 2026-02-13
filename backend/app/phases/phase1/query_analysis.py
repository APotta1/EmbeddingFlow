"""
Task 1.1: Query Analysis
- Intent classification
- Entity extraction
- Detect if time-sensitive
- Complexity assessment
"""
from app.config import Settings
from app.phases.phase1.schemas import (
    QueryAnalysisResult,
    IntentCategory,
    ComplexityLevel,
    ExtractedEntity,
)
from openai import OpenAI
import json


QUERY_ANALYSIS_SYSTEM = """You are a query analysis system for a RAG pipeline. Analyze the user's search query and return structured JSON.

Output JSON with exactly these fields:
- intent: one of factual, comparison, how_to, opinion_or_analysis, recent_events, definition, multi_hop, other
- intent_confidence: float 0-1
- entities: list of { "text": "...", "type": "PERSON|ORG|PRODUCT|EVENT|DATE|LOCATION|OTHER", "relevance": "high|medium|low" }
- time_sensitive: boolean - true if the query asks for recent/latest/current/breaking/today/news or time-bound info
- time_sensitivity_reason: one short sentence
- complexity: one of simple, moderate, complex (simple=single fact, moderate=few entities or one comparison, complex=multi-hop or many sources)
- complexity_reason: one short sentence

Be concise. Extract only entities that appear in the query and are relevant to answering it."""


def _parse_llm_response(raw: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = raw.strip()
    # Remove optional markdown code block
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    return json.loads(text)


def _normalize_enum(value: str, enum_class) -> str:
    """Map LLM string to enum value; default to first enum if invalid."""
    v = value.strip().lower().replace(" ", "_")
    for e in enum_class:
        if e.value == v or e.name.lower() == v:
            return e.value
    return list(enum_class)[0].value


def analyze_query(query: str, settings: Settings | None = None) -> QueryAnalysisResult:
    """
    Run Task 1.1: Query Analysis.
    Returns intent, entities, time_sensitivity, and complexity.
    """
    settings = settings or Settings()
    if not settings.openai_api_key:
        # Fallback: rule-based stub for testing without API key
        return _analyze_query_fallback(query)

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=settings.model_query_analysis,
        messages=[
            {"role": "system", "content": QUERY_ANALYSIS_SYSTEM},
            {"role": "user", "content": f"Analyze this query:\n{query}"},
        ],
        temperature=0.1,
    )
    raw = resp.choices[0].message.content or "{}"
    data = _parse_llm_response(raw)

    # Map to our enums and types
    intent = _normalize_enum(
        data.get("intent", "other"),
        IntentCategory,
    )
    complexity = _normalize_enum(
        data.get("complexity", "simple"),
        ComplexityLevel,
    )
    entities = []
    for e in data.get("entities") or []:
        if isinstance(e, dict) and e.get("text"):
            entities.append(
                ExtractedEntity(
                    text=str(e["text"]),
                    type=str(e.get("type", "OTHER")).upper(),
                    relevance=str(e.get("relevance", "high")).lower(),
                )
            )

    return QueryAnalysisResult(
        intent=IntentCategory(intent),
        intent_confidence=float(data.get("intent_confidence", 1.0)),
        entities=entities,
        time_sensitive=bool(data.get("time_sensitive", False)),
        time_sensitivity_reason=data.get("time_sensitivity_reason") or None,
        complexity=ComplexityLevel(complexity),
        complexity_reason=data.get("complexity_reason") or None,
    )


def _analyze_query_fallback(query: str) -> QueryAnalysisResult:
    """Rule-based fallback when no OpenAI API key is set."""
    q = query.lower()
    # Time-sensitive heuristics
    time_keywords = (
        "latest", "recent", "current", "today", "this week", "this month",
        "breaking", "news", "2024", "2025", "last year", "last month",
    )
    time_sensitive = any(k in q for k in time_keywords)
    # Simple complexity heuristics
    word_count = len(query.split())
    if word_count <= 5 and " vs " not in q and " and " not in q:
        complexity = ComplexityLevel.SIMPLE
        complexity_reason = "Short, single-focus query"
    elif word_count <= 12 and (" vs " in q or " compare " in q or " and " in q):
        complexity = ComplexityLevel.MODERATE
        complexity_reason = "Comparison or multiple concepts"
    else:
        complexity = ComplexityLevel.COMPLEX
        complexity_reason = "Long or multi-part query"
    # Default intent
    if "how " in q and (" do " in q or " to " in q):
        intent = IntentCategory.HOW_TO
    elif " vs " in q or " compare " in q:
        intent = IntentCategory.COMPARISON
    elif time_sensitive:
        intent = IntentCategory.RECENT_EVENTS
    else:
        intent = IntentCategory.FACTUAL

    return QueryAnalysisResult(
        intent=intent,
        intent_confidence=0.7,
        entities=[],  # No entity extraction in fallback
        time_sensitive=time_sensitive,
        time_sensitivity_reason="Keyword-based detection" if time_sensitive else "No recency cues",
        complexity=complexity,
        complexity_reason=complexity_reason,
    )


"""
Pydantic schemas for Phase 1 (Query Processing) and the JSON sent to Phase 2.
"""
from enum import Enum
from pydantic import BaseModel, Field


class IntentCategory(str, Enum):
    """Intent classification for the user query."""

    FACTUAL = "factual"  # What is X? When did Y happen?
    COMPARISON = "comparison"  # Compare A vs B
    HOW_TO = "how_to"  # How do I...?
    OPINION_OR_ANALYSIS = "opinion_or_analysis"  # What do experts say about X?
    RECENT_EVENTS = "recent_events"  # Latest news on X
    DEFINITION = "definition"  # What does X mean?
    MULTI_HOP = "multi_hop"  # Complex, needs multiple steps
    OTHER = "other"


class ComplexityLevel(str, Enum):
    """Complexity assessment."""

    SIMPLE = "simple"  # Single fact, one entity
    MODERATE = "moderate"  # Few entities or one comparison
    COMPLEX = "complex"  # Multi-hop, many entities, comparison across sources


class ExtractedEntity(BaseModel):
    """A single extracted entity from the query."""

    text: str = Field(..., description="Surface form in the query")
    type: str = Field(..., description="e.g. PERSON, ORG, PRODUCT, EVENT, DATE, LOCATION")
    relevance: str = Field(
        default="high",
        description="high | medium | low - relevance to answering the query",
    )


class QueryAnalysisResult(BaseModel):
    """Output of Task 1.1: Query Analysis."""

    intent: IntentCategory = Field(..., description="Classified intent of the query")
    intent_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in intent classification (0-1)",
    )
    entities: list[ExtractedEntity] = Field(
        default_factory=list,
        description="Extracted entities (people, orgs, dates, etc.)",
    )
    time_sensitive: bool = Field(
        ...,
        description="True if the query asks for recent/latest/current information",
    )
    time_sensitivity_reason: str | None = Field(
        default=None,
        description="Short explanation why time-sensitive or not",
    )
    complexity: ComplexityLevel = Field(
        ...,
        description="Assessed complexity of the query",
    )
    complexity_reason: str | None = Field(
        default=None,
        description="Short explanation for complexity level",
    )


# Phase 2 input schema lives in Phase 2 or a shared module once 1.2/1.3 exist.
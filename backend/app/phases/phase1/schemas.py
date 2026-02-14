"""
Phase 1 — Query Processing: Pydantic schemas for Task 1.1 Query Analysis.
"""

from pydantic import BaseModel, Field


class IntentOutput(BaseModel):
    """Intent classification result."""

    primary: str = Field(..., description="Primary intent: factual, comparison, how_to, opinion, news, numerical, other")
    categories: list[str] = Field(default_factory=list, description="All applicable intent categories")


class EntityOutput(BaseModel):
    """Single extracted entity."""

    text: str = Field(..., description="Entity surface form")
    type: str = Field(..., description="Entity type: person, organization, product, event, place, date, topic, other")


class ComplexityOutput(BaseModel):
    """Complexity assessment result."""

    level: str = Field(..., description="simple | moderate | complex")
    suggested_sub_questions: int = Field(default=0, ge=0, description="Suggested number of sub-questions for decomposition")
    multi_hop: bool = Field(default=False, description="Whether query likely requires multi-hop reasoning")


class QueryAnalysisOutput(BaseModel):
    """Full output of Task 1.1 Query Analysis."""

    intent: IntentOutput
    entities: list[EntityOutput] = Field(default_factory=list)
    time_sensitive: bool = False
    time_expressions: list[str] = Field(default_factory=list, description="Normalized time expressions if time_sensitive")
    complexity: ComplexityOutput


class QueryAnalysisResponse(BaseModel):
    """Top-level response containing query_analysis for Phase 2."""

    query_analysis: QueryAnalysisOutput

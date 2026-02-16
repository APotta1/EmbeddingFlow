"""
Phase 1 — Query Processing: Pydantic schemas for Tasks 1.1, 1.2, 1.3.
"""

from pydantic import BaseModel, Field


# ----- Task 1.1: Query Analysis -----


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


# ----- Task 1.2: Query Decomposition -----


class QueryDecompositionOutput(BaseModel):
    """Prioritized sub-questions from Task 1.2."""

    sub_questions: list[str] = Field(
        ...,
        min_length=1,
        description="Sub-questions in priority order (first = highest priority)",
    )


# ----- Task 1.3: Query Expansion -----


class QueryExpansionOutput(BaseModel):
    """Search variations and related terms from Task 1.3."""

    search_variations: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="3-5 search query variations for multi-source search",
    )
    synonyms_and_related: list[str] = Field(
        default_factory=list,
        description="Synonyms and related terms to consider",
    )
    temporal_constraints: list[str] = Field(
        default_factory=list,
        description="Time-related filters if query is time-sensitive (e.g. '2024', 'recent')",
    )


# ----- Phase 1 full output (for Phase 2) -----


class Phase1Output(BaseModel):
    """Complete Phase 1 output: analysis + decomposition + expansion. Sent to Phase 2."""

    query_analysis: QueryAnalysisOutput
    query_decomposition: QueryDecompositionOutput
    query_expansion: QueryExpansionOutput


# ----- Legacy / single-task responses -----


class QueryAnalysisResponse(BaseModel):
    """Top-level response containing query_analysis only (Task 1.1)."""

    query_analysis: QueryAnalysisOutput

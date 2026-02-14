"""Pytest fixtures for Phase 1 tests."""

import pytest

from app.phases.phase1.schemas import (
    ComplexityOutput,
    EntityOutput,
    IntentOutput,
    QueryAnalysisOutput,
    QueryAnalysisResponse,
)


@pytest.fixture
def intent_factual():
    return IntentOutput(primary="factual", categories=["factual", "numerical"])


@pytest.fixture
def intent_comparison():
    return IntentOutput(primary="comparison", categories=["comparison"])


@pytest.fixture
def entities_sample():
    return [
        EntityOutput(text="Tesla", type="organization"),
        EntityOutput(text="Q2 2024", type="date"),
    ]


@pytest.fixture
def complexity_simple():
    return ComplexityOutput(level="simple", suggested_sub_questions=0, multi_hop=False)


@pytest.fixture
def complexity_complex():
    return ComplexityOutput(level="complex", suggested_sub_questions=4, multi_hop=True)


@pytest.fixture
def full_query_analysis_output(intent_factual, entities_sample, complexity_simple):
    return QueryAnalysisOutput(
        intent=intent_factual,
        entities=entities_sample,
        time_sensitive=True,
        time_expressions=["Q2 2024"],
        complexity=complexity_simple,
    )


@pytest.fixture
def full_query_analysis_response(full_query_analysis_output):
    return QueryAnalysisResponse(query_analysis=full_query_analysis_output)


@pytest.fixture
def llm_response_json():
    """Minimal valid JSON string as returned by LLM."""
    return '''{
        "intent": {"primary": "factual", "categories": ["factual"]},
        "entities": [{"text": "OpenAI", "type": "organization"}],
        "time_sensitive": false,
        "time_expressions": [],
        "complexity": {"level": "simple", "suggested_sub_questions": 0, "multi_hop": false}
    }'''


@pytest.fixture
def llm_response_with_markdown(llm_response_json):
    """LLM response wrapped in markdown code block."""
    return "```json\n" + llm_response_json + "\n```"

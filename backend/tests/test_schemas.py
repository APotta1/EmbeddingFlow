"""Tests for Phase 1 Task 1.1 schemas."""

import pytest
from pydantic import ValidationError

from app.phases.phase1.schemas import (
    ComplexityOutput,
    EntityOutput,
    IntentOutput,
    QueryAnalysisOutput,
    QueryAnalysisResponse,
)


class TestIntentOutput:
    """Intent classification schema."""

    def test_valid_intent(self):
        intent = IntentOutput(primary="factual", categories=["factual", "numerical"])
        assert intent.primary == "factual"
        assert intent.categories == ["factual", "numerical"]

    def test_categories_default_empty(self):
        intent = IntentOutput(primary="other")
        assert intent.categories == []

    def test_primary_required(self):
        with pytest.raises(ValidationError):
            IntentOutput(categories=["factual"])

    def test_serialization_roundtrip(self):
        intent = IntentOutput(primary="news", categories=["news", "current_events"])
        data = intent.model_dump()
        restored = IntentOutput.model_validate(data)
        assert restored.primary == intent.primary
        assert restored.categories == intent.categories


class TestEntityOutput:
    """Single entity schema."""

    def test_valid_entity(self):
        entity = EntityOutput(text="Tesla", type="organization")
        assert entity.text == "Tesla"
        assert entity.type == "organization"

    def test_text_required(self):
        with pytest.raises(ValidationError):
            EntityOutput(type="organization")

    def test_type_required(self):
        with pytest.raises(ValidationError):
            EntityOutput(text="Tesla")

    def test_all_entity_types(self):
        for etype in ("person", "organization", "product", "event", "place", "date", "topic", "other"):
            entity = EntityOutput(text="x", type=etype)
            assert entity.type == etype

    def test_json_serialization(self):
        entity = EntityOutput(text="Q2 2024", type="date")
        assert entity.model_dump() == {"text": "Q2 2024", "type": "date"}


class TestComplexityOutput:
    """Complexity assessment schema."""

    def test_valid_simple(self):
        comp = ComplexityOutput(level="simple", suggested_sub_questions=0, multi_hop=False)
        assert comp.level == "simple"
        assert comp.suggested_sub_questions == 0
        assert comp.multi_hop is False

    def test_valid_complex(self):
        comp = ComplexityOutput(level="complex", suggested_sub_questions=4, multi_hop=True)
        assert comp.level == "complex"
        assert comp.suggested_sub_questions == 4
        assert comp.multi_hop is True

    def test_defaults(self):
        comp = ComplexityOutput(level="moderate")
        assert comp.suggested_sub_questions == 0
        assert comp.multi_hop is False

    def test_level_required(self):
        with pytest.raises(ValidationError):
            ComplexityOutput(suggested_sub_questions=1)

    def test_suggested_sub_questions_ge_zero(self):
        with pytest.raises(ValidationError):
            ComplexityOutput(level="simple", suggested_sub_questions=-1)


class TestQueryAnalysisOutput:
    """Full query analysis output schema."""

    def test_valid_full_output(self, intent_factual, entities_sample, complexity_simple):
        out = QueryAnalysisOutput(
            intent=intent_factual,
            entities=entities_sample,
            time_sensitive=True,
            time_expressions=["2024", "Q2"],
            complexity=complexity_simple,
        )
        assert out.intent.primary == "factual"
        assert len(out.entities) == 2
        assert out.time_sensitive is True
        assert out.time_expressions == ["2024", "Q2"]
        assert out.complexity.level == "simple"

    def test_entities_default_empty(self, intent_factual, complexity_simple):
        out = QueryAnalysisOutput(
            intent=intent_factual,
            complexity=complexity_simple,
        )
        assert out.entities == []
        assert out.time_sensitive is False
        assert out.time_expressions == []

    def test_time_expressions_default_empty(self, intent_factual, complexity_simple):
        out = QueryAnalysisOutput(
            intent=intent_factual,
            time_sensitive=False,
            complexity=complexity_simple,
        )
        assert out.time_expressions == []

    def test_intent_required(self, entities_sample, complexity_simple):
        with pytest.raises(ValidationError):
            QueryAnalysisOutput(
                entities=entities_sample,
                complexity=complexity_simple,
            )

    def test_complexity_required(self, intent_factual, entities_sample):
        with pytest.raises(ValidationError):
            QueryAnalysisOutput(
                intent=intent_factual,
                entities=entities_sample,
            )

    def test_model_dump_json_phase2_ready(self, full_query_analysis_output):
        data = full_query_analysis_output.model_dump()
        assert "intent" in data
        assert "entities" in data
        assert "time_sensitive" in data
        assert "time_expressions" in data
        assert "complexity" in data
        assert data["intent"]["primary"] == "factual"
        assert len(data["entities"]) == 2


class TestQueryAnalysisResponse:
    """Top-level response schema for Phase 2."""

    def test_valid_response(self, full_query_analysis_response):
        assert full_query_analysis_response.query_analysis.intent.primary == "factual"
        assert len(full_query_analysis_response.query_analysis.entities) == 2

    def test_query_analysis_required(self):
        with pytest.raises(ValidationError):
            QueryAnalysisResponse()

    def test_json_roundtrip(self, full_query_analysis_response):
        json_str = full_query_analysis_response.model_dump_json()
        restored = QueryAnalysisResponse.model_validate_json(json_str)
        assert restored.query_analysis.intent.primary == full_query_analysis_response.query_analysis.intent.primary
        assert len(restored.query_analysis.entities) == len(full_query_analysis_response.query_analysis.entities)

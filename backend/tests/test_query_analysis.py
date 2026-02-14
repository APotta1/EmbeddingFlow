"""Tests for Phase 1 Task 1.1 query_analysis module."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from app.phases.phase1.query_analysis import (
    _parse_llm_response,
    analyze_query,
)
from app.phases.phase1.schemas import (
    ComplexityOutput,
    EntityOutput,
    IntentOutput,
    QueryAnalysisOutput,
    QueryAnalysisResponse,
)


class TestParseLlmResponse:
    """_parse_llm_response behavior."""

    def test_plain_json(self, llm_response_json):
        result = _parse_llm_response(llm_response_json)
        assert isinstance(result, dict)
        assert result["intent"]["primary"] == "factual"
        assert result["time_sensitive"] is False
        assert result["complexity"]["level"] == "simple"

    def test_json_with_markdown_code_block(self, llm_response_with_markdown):
        result = _parse_llm_response(llm_response_with_markdown)
        assert result["intent"]["primary"] == "factual"
        assert "entities" in result

    def test_json_with_markdown_no_lang_tag(self, llm_response_json):
        wrapped = "```\n" + llm_response_json + "\n```"
        result = _parse_llm_response(wrapped)
        assert result["intent"]["primary"] == "factual"

    def test_whitespace_stripped(self, llm_response_json):
        result = _parse_llm_response("  \n  " + llm_response_json + "  \n  ")
        assert result["intent"]["primary"] == "factual"

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_llm_response("not json at all")

    def test_empty_string_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_llm_response("")


class TestAnalyzeQueryEmpty:
    """analyze_query with empty or whitespace-only query (no LLM call)."""

    def test_empty_string(self):
        result = analyze_query("")
        assert isinstance(result, QueryAnalysisResponse)
        assert result.query_analysis.intent.primary == "other"
        assert result.query_analysis.intent.categories == []
        assert result.query_analysis.entities == []
        assert result.query_analysis.time_sensitive is False
        assert result.query_analysis.time_expressions == []
        assert result.query_analysis.complexity.level == "simple"
        assert result.query_analysis.complexity.suggested_sub_questions == 0
        assert result.query_analysis.complexity.multi_hop is False

    def test_whitespace_only(self):
        result = analyze_query("   \n\t  ")
        assert result.query_analysis.intent.primary == "other"
        assert result.query_analysis.entities == []

    def test_none_equivalent_empty(self):
        # Function receives "" so we don't pass None from API; test empty is enough
        result = analyze_query("")
        assert result.query_analysis.complexity.level == "simple"


class TestAnalyzeQueryWithMockedClient:
    """analyze_query with mocked OpenAI client."""

    @pytest.fixture(autouse=True)
    def set_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def _make_mock_response(self, content: str):
        choice = MagicMock()
        choice.message.content = content
        choice.message.role = "assistant"
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_returns_structured_response(self, llm_response_json, set_api_key):
        with patch("app.phases.phase1.query_analysis.OpenAI") as mock_openai:
            client_instance = MagicMock()
            client_instance.chat.completions.create.return_value = self._make_mock_response(llm_response_json)
            mock_openai.return_value = client_instance

            result = analyze_query("What is OpenAI?")

        assert isinstance(result, QueryAnalysisResponse)
        assert result.query_analysis.intent.primary == "factual"
        assert result.query_analysis.time_sensitive is False
        assert len(result.query_analysis.entities) == 1
        assert result.query_analysis.entities[0].text == "OpenAI"
        assert result.query_analysis.entities[0].type == "organization"

    def test_handles_markdown_wrapped_response(self, llm_response_with_markdown, set_api_key):
        with patch("app.phases.phase1.query_analysis.OpenAI") as mock_openai:
            client_instance = MagicMock()
            client_instance.chat.completions.create.return_value = self._make_mock_response(llm_response_with_markdown)
            mock_openai.return_value = client_instance

            result = analyze_query("Tell me about Tesla")

        assert result.query_analysis.intent.primary == "factual"
        assert result.query_analysis.entities[0].text == "OpenAI"

    def test_maps_partial_llm_response(self, set_api_key):
        # LLM returns missing or null fields — we use defaults
        minimal = json.dumps({
            "intent": {"primary": "news"},
            "entities": [{"text": "Earthquake"}],
            "time_sensitive": True,
            "time_expressions": ["recent"],
            "complexity": {},
        })
        with patch("app.phases.phase1.query_analysis.OpenAI") as mock_openai:
            client_instance = MagicMock()
            client_instance.chat.completions.create.return_value = self._make_mock_response(minimal)
            mock_openai.return_value = client_instance

            result = analyze_query("Recent earthquake news")

        assert result.query_analysis.intent.primary == "news"
        assert result.query_analysis.intent.categories == []
        assert result.query_analysis.entities[0].text == "Earthquake"
        assert result.query_analysis.entities[0].type == "other"
        assert result.query_analysis.time_sensitive is True
        assert result.query_analysis.time_expressions == ["recent"]
        assert result.query_analysis.complexity.level == "simple"
        assert result.query_analysis.complexity.suggested_sub_questions == 0
        assert result.query_analysis.complexity.multi_hop is False

    def test_strips_query_whitespace(self, llm_response_json, set_api_key):
        with patch("app.phases.phase1.query_analysis.OpenAI") as mock_openai:
            client_instance = MagicMock()
            client_instance.chat.completions.create.return_value = self._make_mock_response(llm_response_json)
            mock_openai.return_value = client_instance

            analyze_query("  what is AI  ")

        call_kw = client_instance.chat.completions.create.call_args[1]
        user_content = next(m["content"] for m in call_kw["messages"] if m["role"] == "user")
        assert "what is AI" in user_content
        assert user_content.strip().endswith("what is AI")


class TestGetClientMissingApiKey:
    """_get_client raises when OPENAI_API_KEY is missing."""

    def test_analyze_query_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "")
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            analyze_query("non-empty query")

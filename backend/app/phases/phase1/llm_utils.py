"""Shared LLM helpers for Phase 1 (OpenAI client, JSON parsing)."""

import json
import os
from typing import Any

from openai import OpenAI


def get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    return OpenAI(api_key=api_key)


def parse_llm_response(content: str) -> dict[str, Any]:
    """Parse JSON from LLM response, stripping markdown code blocks if present."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return json.loads(text)


def get_model() -> str:
    return os.environ.get("OPENAI_QUERY_ANALYSIS_MODEL", "gpt-4o-mini")

"""Shared LLM helpers for Phase 1 (Groq SDK, JSON parsing)."""

import json
import os
from typing import Any

from groq import Groq


def get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is required")
    return Groq(api_key=api_key)


def parse_llm_response(content: str | None) -> dict[str, Any]:
    """Parse JSON from LLM response, stripping markdown code blocks if present."""
    if content is None or not str(content).strip():
        return {}
    text = str(content).strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def get_model() -> str:
    return os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

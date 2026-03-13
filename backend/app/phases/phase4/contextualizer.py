
from __future__ import annotations

from typing import Iterable, List, Optional

from app.phases.phase1.llm_utils import get_client, get_model
from app.phases.phase3.schemas import ExtractedDocument
from app.phases.phase4.schemas import Chunk, Phase4Config  # type: ignore[import-not-found]


CONTEXT_SYSTEM_PROMPT = """You add concise context to text chunks from documents.

For each chunk, you will be given:
- The original user query
- Document metadata (title, domain/source, optional publish date)
- The chunk text itself

Your job:
1. Briefly (in 1–2 short sentences) situate the chunk within the overall document:
   - What is this section about?
   - How does it relate to the document's topic?
   - Mention key entities or time period only if clearly helpful.
2. Then output the original chunk text unchanged after the context line.

Format your response as plain text:
[Context: ...]
<original chunk text>

Keep the context line under 200 characters. Do not add explanations outside this format."""


def _build_user_prompt(
    original_query: str,
    doc: ExtractedDocument,
    chunk: Chunk,
) -> str:
    """Construct the user message for the contextualization prompt."""

    lines: List[str] = []
    lines.append(f"User query: {original_query.strip()}")
    lines.append("")
    lines.append("Document metadata:")
    lines.append(f"- Title: {doc.title or '(unknown)'}")
    lines.append(f"- URL: {doc.url}")
    lines.append(f"- Domain: {doc.domain or '(unknown)'}")
    if doc.publish_date:
        lines.append(f"- Publish date: {doc.publish_date}")
    lines.append("")
    lines.append("Chunk text:")
    lines.append(chunk.raw_text)
    return "\n".join(lines)


def contextualize_chunk(
    chunk: Chunk,
    doc: ExtractedDocument,
    original_query: str,
    config: Phase4Config,
) -> Chunk:
    """
    Contextualize a single chunk.

    If contextualization is disabled or the LLM call fails, we return the chunk
    unchanged (contextualized_text == raw_text).
    """

    if not config.enable_contextualization:
        return chunk

    try:
        client = get_client()
        model = config.llm_model or get_model()
        user_content = _build_user_prompt(original_query, doc, chunk)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CONTEXT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            return chunk
        return Chunk(**{**chunk.model_dump(), "contextualized_text": content})
    except Exception:
        # Fail-safe: keep raw_text as contextualized_text if anything goes wrong.
        return chunk


def contextualize_chunks_for_document(
    doc: ExtractedDocument,
    chunks: List[Chunk],
    original_query: str,
    config: Phase4Config,
) -> List[Chunk]:
    """
    Contextualize all chunks belonging to a single document.

    This is a simple sequential implementation; batching / caching can be added
    later if needed.
    """

    out: List[Chunk] = []
    for chunk in chunks:
        out.append(contextualize_chunk(chunk, doc, original_query, config))
    return out


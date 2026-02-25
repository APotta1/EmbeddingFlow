"""
Phase 3 — Content Extraction & Cleaning.

Consumes Phase 2 output (ranked URLs) and produces cleaned documents
ready for contextual chunking (Phase 4).
"""

from pydantic import BaseModel, Field
from typing import Optional

from app.phases.phase2.schemas import SearchResult


class ExtractedDocument(BaseModel):
    """Single extracted and cleaned document."""

    url: str
    title: Optional[str] = Field(default=None, description="Document title if available")
    author: Optional[str] = Field(default=None, description="Author if available")
    publish_date: Optional[str] = Field(
        default=None, description="Publish date as ISO string or raw date if available"
    )
    domain: Optional[str] = Field(default=None, description="Domain extracted from URL")
    source_api: Optional[str] = Field(default=None, description="Which search API produced this URL")
    position: Optional[int] = Field(default=None, description="Original rank/position in search results")

    content: str = Field(..., description="Main article body, cleaned")
    content_paragraphs: list[str] = Field(
        default_factory=list,
        description="Paragraph-level segmentation of content (in order)",
    )

    raw_metadata: dict = Field(
        default_factory=dict,
        description="Optional raw metadata from extractors (debugging / future use)",
    )


class Phase3Stats(BaseModel):
    """Diagnostics for Phase 3."""

    total_input_urls: int
    fetched: int
    successful: int
    skipped_robots: int
    failed: int
    below_quality_threshold: int


class Phase3Output(BaseModel):
    """Complete Phase 3 output: cleaned documents + stats."""

    original_query: str
    documents: list[ExtractedDocument]
    stats: Phase3Stats
    # Keep track of which URLs we actually tried (for debugging / audits)
    input_urls: list[SearchResult]


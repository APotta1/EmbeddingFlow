from typing import Optional

from pydantic import BaseModel, Field

from app.phases.phase3.schemas import ExtractedDocument, Phase3Output


class Phase4Config(BaseModel):
    """
    Configuration for Phase 4 chunking and contextual enrichment.

    The size parameters are in "token-equivalent" units. We approximate token
    counts using word counts unless a downstream tokenizer is plugged in.
    """

    # Target chunk size (approximate tokens)
    min_chunk_tokens: int = Field(
        default=400,
        description="Minimum approximate tokens per chunk before we consider it complete.",
    )
    max_chunk_tokens: int = Field(
        default=800,
        description="Maximum approximate tokens per chunk.",
    )
    overlap_tokens: int = Field(
        default=160,
        description="Approximate token overlap between consecutive chunks.",
    )

    # Semantic boundary tuning (lexical similarity between adjacent paragraphs)
    boundary_model: str = Field(
        default="tfidf",
        description="Boundary model for detecting topic shifts: 'tfidf' (default) or 'jaccard'.",
    )
    min_paragraph_similarity: float = Field(
        default=0.1,
        description=(
            "If lexical similarity between adjacent paragraphs falls below this, "
            "it's treated as a stronger boundary (we avoid crossing unless needed "
            "to satisfy min_chunk_tokens). Range 0–1."
        ),
    )

    # Contextual enrichment
    enable_contextualization: bool = Field(
        default=True,
        description="If False, contextualized_text will equal raw_text.",
    )
    llm_model: Optional[str] = Field(
        default=None,
        description="Optional override for the LLM model used for contextualization.",
    )


class Chunk(BaseModel):
    """
    Single chunk derived from an ExtractedDocument.

    raw_text: the original slice of the document used for retrieval.
    contextualized_text: the LLM-enriched version with document/section context.
    """

    # Identity / traceability
    document_index: int = Field(
        ...,
        description="Index of the parent document in Phase3Output.documents.",
    )
    chunk_index: int = Field(
        ...,
        description="Index of this chunk within the parent document (0-based).",
    )

    # Source document metadata (duplicated here for convenience at query time)
    url: str
    title: Optional[str] = None
    domain: Optional[str] = None
    source_api: Optional[str] = None
    publish_date: Optional[str] = None

    # Span information (paragraph-based for now)
    start_paragraph_index: int = Field(
        ...,
        description="Inclusive start index into ExtractedDocument.content_paragraphs.",
    )
    end_paragraph_index: int = Field(
        ...,
        description="Exclusive end index into ExtractedDocument.content_paragraphs.",
    )

    # Text content
    raw_text: str = Field(
        ...,
        description="Concatenation of the underlying paragraph span.",
    )
    contextualized_text: str = Field(
        ...,
        description="Raw text plus LLM-added document/section context.",
    )

    # Diagnostics
    approx_token_count: int = Field(
        ...,
        description=(
            "Approximate token count for this chunk (used for monitoring and tuning)."
        ),
    )


class Phase4Stats(BaseModel):
    """Diagnostics for Phase 4."""

    total_documents: int
    total_chunks: int
    contextualized_chunks: int
    failed_contextualizations: int


class Phase4Output(BaseModel):
    """
    Complete Phase 4 output: original query, documents (as in Phase 3),
    and contextualized chunks ready for embedding.
    """

    original_query: str
    documents: list[ExtractedDocument]
    chunks: list[Chunk]
    stats: Phase4Stats

    # Keep a reference to the Phase 3 input for debugging / audits.
    phase3_input: Phase3Output


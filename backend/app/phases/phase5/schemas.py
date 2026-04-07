from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from app.phases.phase4.schemas import Chunk, Phase4Output

EmbeddingProvider = Literal["openai", "voyage", "cohere"]
VectorStoreType = Literal["faiss", "qdrant", "pinecone"]


class EmbeddingMetadata(BaseModel):
    """Metadata persisted alongside each embedding vector."""

    original_chunk_text: str
    contextualized_chunk_text: str
    source_url: str
    document_title: Optional[str] = None
    publish_date: Optional[str] = None
    chunk_position: int = Field(
        ...,
        description="Chunk index inside the source document (0-based).",
    )
    document_index: int = Field(
        ...,
        description="Document index from Phase 4 output.",
    )
    start_paragraph_index: int
    end_paragraph_index: int
    approx_token_count: int
    source_api: Optional[str] = None
    domain: Optional[str] = None


class EmbeddingRecord(BaseModel):
    """Single vectorized chunk."""

    id: str
    text: str
    embedding: list[float]
    metadata: EmbeddingMetadata


class Phase5Config(BaseModel):
    """
    Config for embedding generation and vector indexing.

    For provider credentials, use:
    - OPENAI_API_KEY
    - VOYAGE_API_KEY
    - COHERE_API_KEY
    """

    embedding_provider: EmbeddingProvider = "openai"
    embedding_model: str = "text-embedding-3-large"
    batch_size: int = Field(default=32, ge=1, le=256)

    vector_store: VectorStoreType = "faiss"
    collection_name: str = "embeddingflow_chunks"

    # FAISS options
    faiss_metric: Literal["cosine", "l2"] = "cosine"

    # Qdrant options
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None

    # Pinecone options
    pinecone_api_key: Optional[str] = None
    pinecone_index_name: Optional[str] = None
    pinecone_host: Optional[str] = None


class Phase5Stats(BaseModel):
    total_chunks_in: int
    embedded_chunks: int
    stored_vectors: int
    failed_embeddings: int
    provider: EmbeddingProvider
    model: str
    vector_store: VectorStoreType
    embedding_dimensions: int = 0


class Phase5Output(BaseModel):
    original_query: str
    records: list[EmbeddingRecord]
    stats: Phase5Stats
    phase4_input: Phase4Output
    store_response: dict[str, Any] = Field(default_factory=dict)


def chunk_to_metadata(chunk: Chunk) -> EmbeddingMetadata:
    return EmbeddingMetadata(
        original_chunk_text=chunk.raw_text,
        contextualized_chunk_text=chunk.contextualized_text,
        source_url=chunk.url,
        document_title=chunk.title,
        publish_date=chunk.publish_date,
        chunk_position=chunk.chunk_index,
        document_index=chunk.document_index,
        start_paragraph_index=chunk.start_paragraph_index,
        end_paragraph_index=chunk.end_paragraph_index,
        approx_token_count=chunk.approx_token_count,
        source_api=chunk.source_api,
        domain=chunk.domain,
    )

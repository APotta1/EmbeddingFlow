from __future__ import annotations

from typing import Optional
from uuid import uuid4

from app.phases.phase4.schemas import Phase4Output
from app.phases.phase5.embeddings import embed_texts
from app.phases.phase5.schemas import (
    EmbeddingRecord,
    Phase5Config,
    Phase5Output,
    Phase5Stats,
    chunk_to_metadata,
)
from app.phases.phase5.vector_store import get_vector_store


def run_phase5(
    phase4_output: Phase4Output,
    config: Optional[Phase5Config] = None,
) -> Phase5Output:
    """
    Phase 5: embedding generation + vector indexing.

    Input: contextualized chunks from Phase 4.
    Output: persisted vectors with rich metadata.
    """
    cfg = config or Phase5Config()

    if not phase4_output.chunks:
        return Phase5Output(
            original_query=phase4_output.original_query,
            records=[],
            stats=Phase5Stats(
                total_chunks_in=0,
                embedded_chunks=0,
                stored_vectors=0,
                failed_embeddings=0,
                provider=cfg.embedding_provider,
                model=cfg.embedding_model,
                vector_store=cfg.vector_store,
                embedding_dimensions=0,
            ),
            phase4_input=phase4_output,
            store_response={},
        )

    texts = [c.contextualized_text for c in phase4_output.chunks]
    vectors, batch_diag = embed_texts(
        texts,
        provider=cfg.embedding_provider,
        model=cfg.embedding_model,
        batch_size=cfg.batch_size,
    )

    limit = min(len(phase4_output.chunks), len(vectors))
    records: list[EmbeddingRecord] = []
    for i in range(limit):
        chunk = phase4_output.chunks[i]
        records.append(
            EmbeddingRecord(
                id=str(uuid4()),
                text=chunk.contextualized_text,
                embedding=vectors[i],
                metadata=chunk_to_metadata(chunk),
            )
        )

    store = get_vector_store(cfg)
    store_response = store.upsert(records)

    stats = Phase5Stats(
        total_chunks_in=len(phase4_output.chunks),
        embedded_chunks=len(records),
        stored_vectors=int(store_response.get("stored", len(records))),
        failed_embeddings=max(0, len(phase4_output.chunks) - len(records)),
        provider=cfg.embedding_provider,
        model=cfg.embedding_model,
        vector_store=cfg.vector_store,
        embedding_dimensions=len(records[0].embedding) if records else 0,
    )

    return Phase5Output(
        original_query=phase4_output.original_query,
        records=records,
        stats=stats,
        phase4_input=phase4_output,
        store_response={
            "vector_store": cfg.vector_store,
            "collection_name": cfg.collection_name,
            "batch_diagnostics": batch_diag,
            "details": store_response,
        },
    )

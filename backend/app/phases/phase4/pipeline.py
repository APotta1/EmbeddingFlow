

from __future__ import annotations

from typing import Optional

from app.phases.phase3.schemas import Phase3Output
from app.phases.phase4.chunking import (  # type: ignore[import-not-found]
    chunk_document,
)
from app.phases.phase4.contextualizer import (  # type: ignore[import-not-found]
    contextualize_chunks_for_document,
)
from app.phases.phase4.schemas import (  # type: ignore[import-not-found]
    Chunk,
    Phase4Config,
    Phase4Output,
    Phase4Stats,
)


def run_phase4(
    phase3_output: Phase3Output,
    config: Optional[Phase4Config] = None,
) -> Phase4Output:
    """
    Run full Phase 4: semantic chunking + contextual enrichment.

    Input: Phase3Output (documents + stats).
    Output: Phase4Output with contextualized chunks ready for embeddings.
    """

    cfg = config or Phase4Config()

    if not phase3_output.documents:
        return Phase4Output(
            original_query=phase3_output.original_query,
            documents=[],
            chunks=[],
            stats=Phase4Stats(
                total_documents=0,
                total_chunks=0,
                contextualized_chunks=0,
                failed_contextualizations=0,
            ),
            phase3_input=phase3_output,
        )

    all_chunks: list[Chunk] = []
    contextualized_count = 0

    for doc_index, doc in enumerate(phase3_output.documents):
        # 1) Chunk this document
        chunks = chunk_document(doc, doc_index, cfg)
        if not chunks:
            continue

        # 2) Contextualize chunks for this document
        contextualized_chunks = contextualize_chunks_for_document(
            doc,
            chunks,
            original_query=phase3_output.original_query,
            config=cfg,
        )
        all_chunks.extend(contextualized_chunks)
        contextualized_count += sum(
            1 for c in contextualized_chunks if c.contextualized_text != c.raw_text
        )

    stats = Phase4Stats(
        total_documents=len(phase3_output.documents),
        total_chunks=len(all_chunks),
        contextualized_chunks=contextualized_count,
        failed_contextualizations=max(0, len(all_chunks) - contextualized_count)
        if cfg.enable_contextualization
        else 0,
    )

    return Phase4Output(
        original_query=phase3_output.original_query,
        documents=phase3_output.documents,
        chunks=all_chunks,
        stats=stats,
        phase3_input=phase3_output,
    )


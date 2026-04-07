"""Phase 5: Embedding generation and vector indexing."""

from app.phases.phase5.pipeline import run_phase5
from app.phases.phase5.schemas import (
    EmbeddingMetadata,
    EmbeddingRecord,
    Phase5Config,
    Phase5Output,
    Phase5Stats,
)

__all__ = [
    "run_phase5",
    "EmbeddingMetadata",
    "EmbeddingRecord",
    "Phase5Config",
    "Phase5Output",
    "Phase5Stats",
]

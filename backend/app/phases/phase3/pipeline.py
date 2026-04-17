"""
Phase 3 pipeline: content extraction & cleaning.

Consumes Phase 2 output (Phase2Output) and returns Phase 3 output
(Phase3Output) with cleaned documents and stats.
"""

import asyncio
from typing import Optional

from app.phases.phase2.schemas import Phase2Output
from app.phases.phase3.content_extractor import (
    Phase3Config,
    extract_documents_from_urls,
)
from app.phases.phase3.schemas import Phase3Output


async def run_phase3(
    phase2_output: Phase2Output,
    config: Optional[Phase3Config] = None,
) -> Phase3Output:
    """Async entry: fetch → extract → clean."""
    return await extract_documents_from_urls(
        phase2_output.urls,
        original_query=phase2_output.original_query,
        config=config,
    )


def run_phase3_sync(
    phase2_output: Phase2Output,
    config: Optional[Phase3Config] = None,
) -> Phase3Output:
    """Sync wrapper for CLI scripts and testing. Do not call from async context."""
    return asyncio.run(run_phase3(phase2_output, config))

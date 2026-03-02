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


def run_phase3(
    phase2_output: Phase2Output,
    config: Optional[Phase3Config] = None,
) -> Phase3Output:
    """
    Run full Phase 3 synchronously: fetch → extract → clean.

    This is a thin synchronous wrapper around the async extractor to make it easy
    to call from FastAPI or scripts.
    """

    async def _run() -> Phase3Output:
        return await extract_documents_from_urls(
            phase2_output.urls,
            original_query=phase2_output.original_query,
            config=config,
        )

    # If there is no running loop, use asyncio.run (simple CLI / script case).
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run())

    # If we're already inside an event loop (e.g. FastAPI, Jupyter), create and
    # use a dedicated new loop for this blocking call.
    new_loop = asyncio.new_event_loop()
    try:
        return new_loop.run_until_complete(_run())
    finally:
        new_loop.close()


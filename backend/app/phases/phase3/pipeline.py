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

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        # In case we are inside an async context (e.g. FastAPI), create a task
        # and run it via asyncio.run_until_complete on a new loop.
        return asyncio.run(_run())
    return loop.run_until_complete(_run())


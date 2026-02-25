"""
Phase 3 core: fetch HTML/PDF, extract main content, and clean it.

Implements:
- Task 3.1 Web Scraping (HTML + PDF, metadata, boilerplate removal)
- Task 3.2 Content Cleaning (whitespace, dedupe, quality filter)
- Task 3.3 Rate Limiting (max concurrent requests, simple backoff, robots.txt)
"""

import asyncio
import re
import time
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from typing import Optional
from urllib.parse import urlparse
from urllib import robotparser

import httpx
import trafilatura
from pypdf import PdfReader

from app.phases.phase2.schemas import SearchResult
from app.phases.phase3.schemas import ExtractedDocument, Phase3Output, Phase3Stats


@dataclass
class Phase3Config:
    max_concurrent_requests: int = 8
    request_timeout_seconds: float = 15.0
    max_retries: int = 3
    backoff_base_seconds: float = 0.5
    min_word_count: int = 200
    user_agent: str = "EmbeddingFlowBot/0.1 (+https://github.com/APotta1/EmbeddingFlow)"


_robots_cache: dict[str, robotparser.RobotFileParser] = {}
_robots_lock = asyncio.Lock()


async def _get_robots_parser(domain: str, scheme: str = "https") -> robotparser.RobotFileParser:
    key = f"{scheme}://{domain}"
    async with _robots_lock:
        if key in _robots_cache:
            return _robots_cache[key]
        rp = robotparser.RobotFileParser()
        robots_url = f"{scheme}://{domain}/robots.txt"
        try:
            rp.set_url(robots_url)
            rp.read()
        except Exception:
            # If robots.txt cannot be read, default to allowing
            rp.allow_all = True  # type: ignore[attr-defined]
        _robots_cache[key] = rp
        return rp


async def _respect_robots(url: str, user_agent: str) -> bool:
    parsed = urlparse(url)
    if not parsed.netloc:
        return True
    rp = await _get_robots_parser(parsed.netloc, parsed.scheme or "https")
    try:
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True


def _is_pdf(url: str, content_type: Optional[str]) -> bool:
    if content_type and "pdf" in content_type.lower():
        return True
    return url.lower().endswith(".pdf")


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_paragraphs(text: str) -> list[str]:
    # Simple paragraph split on double newlines or periods with line breaks
    paragraphs = re.split(r"\n{2,}", text)
    cleaned = [p.strip() for p in paragraphs if p.strip()]
    return cleaned


def _hash_content(text: str) -> str:
    return sha256(text.encode("utf-8", errors="ignore")).hexdigest()


async def _fetch_with_retries(
    client: httpx.AsyncClient,
    url: str,
    config: Phase3Config,
) -> Optional[httpx.Response]:
    for attempt in range(config.max_retries):
        try:
            resp = await client.get(url, timeout=config.request_timeout_seconds)
            if resp.status_code >= 500 or resp.status_code in (429,):
                # backoff and retry
                delay = min(config.backoff_base_seconds * (2**attempt), 10.0)
                await asyncio.sleep(delay)
                continue
            if resp.status_code >= 400:
                return None
            return resp
        except (httpx.RequestError, httpx.HTTPStatusError):
            delay = min(config.backoff_base_seconds * (2**attempt), 10.0)
            await asyncio.sleep(delay)
    return None


def _extract_from_pdf(content: bytes, url: str, search_result: SearchResult) -> Optional[ExtractedDocument]:
    try:
        reader = PdfReader(BytesIO(content))
        text_chunks: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text:
                text_chunks.append(page_text)
        full_text = "\n\n".join(text_chunks).strip()
        if not full_text:
            return None
        normalized = _normalize_whitespace(full_text)
        paragraphs = _split_paragraphs(full_text)
        domain = urlparse(url).netloc or search_result.domain
        meta = reader.metadata or {}
        title = getattr(meta, "title", None) or search_result.title
        author = getattr(meta, "author", None)
        return ExtractedDocument(
            url=url,
            title=title,
            author=author,
            publish_date=None,
            domain=domain,
            source_api=search_result.source_api,
            position=search_result.position,
            content=normalized,
            content_paragraphs=paragraphs,
            raw_metadata={"pdf_metadata": {k: str(v) for k, v in dict(meta).items()} if meta else {}},
        )
    except Exception:
        return None


def _extract_from_html(html: str, url: str, search_result: SearchResult) -> Optional[ExtractedDocument]:
    # Use trafilatura to get main content + metadata
    try:
        extracted_json = trafilatura.extract(html, url=url, output_format="json", include_comments=False)
    except Exception:
        extracted_json = None
    text: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    raw_meta: dict = {}

    if extracted_json:
        try:
            data = trafilatura.utils.json_to_dict(extracted_json)  # type: ignore[attr-defined]
        except Exception:
            # Fallback: treat JSON string as plain text
            data = {}
        text = data.get("text")
        title = data.get("title")
        author = data.get("author")
        date = data.get("date") or data.get("date_publish")
        raw_meta = data

    if not text:
        # Fallback: plain text extraction
        try:
            text = trafilatura.extract(html, url=url, include_comments=False)
        except Exception:
            text = None

    if not text:
        return None

    full_text = text.strip()
    normalized = _normalize_whitespace(full_text)
    paragraphs = _split_paragraphs(full_text)
    domain = urlparse(url).netloc or search_result.domain

    if not title:
        title = search_result.title

    return ExtractedDocument(
        url=url,
        title=title,
        author=author,
        publish_date=date,
        domain=domain,
        source_api=search_result.source_api,
        position=search_result.position,
        content=normalized,
        content_paragraphs=paragraphs,
        raw_metadata=raw_meta,
    )


async def extract_documents_from_urls(
    search_results: list[SearchResult],
    original_query: str,
    config: Optional[Phase3Config] = None,
) -> Phase3Output:
    """
    Entry point for Phase 3.

    Args:
        search_results: Ranked URLs from Phase 2.
        original_query: Original user query.
        config: Optional Phase3Config overrides.
    """
    cfg = config or Phase3Config()

    if not search_results:
        return Phase3Output(
            original_query=original_query,
            documents=[],
            stats=Phase3Stats(
                total_input_urls=0,
                fetched=0,
                successful=0,
                skipped_robots=0,
                failed=0,
                below_quality_threshold=0,
            ),
            input_urls=[],
        )

    semaphore = asyncio.Semaphore(cfg.max_concurrent_requests)
    seen_hashes: set[str] = set()
    documents: list[ExtractedDocument] = []
    skipped_robots = 0
    failed = 0
    below_threshold = 0
    fetched = 0

    async def worker(sr: SearchResult):
        nonlocal skipped_robots, failed, below_threshold, fetched

        async with semaphore:
            allowed = await _respect_robots(sr.url, cfg.user_agent)
            if not allowed:
                skipped_robots += 1
                return

            async with httpx.AsyncClient(headers={"User-Agent": cfg.user_agent}, follow_redirects=True) as client:
                resp = await _fetch_with_retries(client, sr.url, cfg)
                if resp is None or resp.content is None:
                    failed += 1
                    return

                fetched += 1
                content_type = resp.headers.get("Content-Type", "")
                doc: Optional[ExtractedDocument]
                if _is_pdf(sr.url, content_type):
                    doc = _extract_from_pdf(resp.content, sr.url, sr)
                else:
                    try:
                        html_text = resp.text
                    except UnicodeDecodeError:
                        html_text = resp.content.decode("utf-8", errors="ignore")
                    doc = _extract_from_html(html_text, sr.url, sr)

                if not doc:
                    failed += 1
                    return

                # Quality filter
                word_count = len(doc.content.split())
                if word_count < cfg.min_word_count:
                    below_threshold += 1
                    return

                # Deduplicate by content hash
                h = _hash_content(doc.content)
                if h in seen_hashes:
                    return
                seen_hashes.add(h)
                documents.append(doc)

    tasks = [worker(sr) for sr in search_results]
    # Run tasks with backpressure
    await asyncio.gather(*tasks)

    stats = Phase3Stats(
        total_input_urls=len(search_results),
        fetched=fetched,
        successful=len(documents),
        skipped_robots=skipped_robots,
        failed=failed,
        below_quality_threshold=below_threshold,
    )

    return Phase3Output(
        original_query=original_query,
        documents=documents,
        stats=stats,
        input_urls=search_results,
    )


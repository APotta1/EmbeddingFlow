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
    robots_timeout_seconds: float = 5.0
    max_retries: int = 3
    backoff_base_seconds: float = 0.5
    min_word_count: int = 200
    user_agent: str = "EmbeddingFlowBot/0.1 (+https://github.com/APotta1/EmbeddingFlow)"
    enable_browser_fallback: bool = False


_robots_cache: dict[str, robotparser.RobotFileParser] = {}
_robots_lock = asyncio.Lock()


async def _get_robots_parser(
    domain: str,
    scheme: str = "https",
    *,
    timeout: float = 5.0,
) -> robotparser.RobotFileParser:
    key = f"{scheme}://{domain}"
    async with _robots_lock:
        if key in _robots_cache:
            return _robots_cache[key]

        rp = robotparser.RobotFileParser()
        robots_url = f"{scheme}://{domain}/robots.txt"
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                resp = await client.get(robots_url, timeout=timeout)
            if resp.status_code < 400 and resp.text:
                rp.parse(resp.text.splitlines())
            else:
                # If robots.txt cannot be read or is missing, default to allowing
                rp.allow_all = True  # type: ignore[attr-defined]
        except Exception:
            # Network errors or timeouts: default to allowing
            rp.allow_all = True  # type: ignore[attr-defined]

        _robots_cache[key] = rp
        return rp


async def _respect_robots(url: str, user_agent: str, *, timeout: float) -> bool:
    parsed = urlparse(url)
    if not parsed.netloc:
        return True
    rp = await _get_robots_parser(parsed.netloc, parsed.scheme or "https", timeout=timeout)
    try:
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True


def _is_pdf(url: str, content_type: Optional[str]) -> bool:
    if content_type and "pdf" in content_type.lower():
        return True
    return url.lower().endswith(".pdf")


def _is_nontext_content_type(content_type: str) -> bool:
    ct = (content_type or "").lower()
    if not ct:
        return False
    if ct.startswith("video/") or ct.startswith("audio/") or ct.startswith("image/"):
        return True
    return False


async def _fetch_with_playwright(url: str, user_agent: str, timeout_seconds: float) -> Optional[str]:
    """
    Optional JS-rendered fetch using Playwright.

    Only used if Phase3Config.enable_browser_fallback=True and Playwright is installed.
    Returns rendered page HTML, or None on failure.
    """

    try:
        from playwright.async_api import async_playwright  # type: ignore[import-not-found]
    except Exception:
        return None

    timeout_ms = int(max(1.0, timeout_seconds) * 1000)
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            try:
                page = await browser.new_page(user_agent=user_agent)
                await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                html = await page.content()
                return html
            finally:
                await browser.close()
    except Exception:
        return None


def _normalize_whitespace(text: str) -> str:
    # Preserve paragraph boundaries (double newlines) while collapsing internal whitespace.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse spaces and tabs but keep newlines
    text = re.sub(r"[^\S\n]+", " ", text)
    # Normalize runs of 3+ newlines down to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_paragraphs(text: str) -> list[str]:
    # Simple paragraph split on double newlines; single newlines stay within a paragraph.
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
            raw_metadata={
                "pdf_metadata": {k: str(v) for k, v in dict(meta).items()} if meta else {}
            },
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
        # Keep metadata but avoid unbounded growth by shallow-copying and stringifying values.
        raw_meta = {}
        total_chars = 0
        for k, v in list(data.items())[:64]:
            key = str(k)
            val = str(v)
            if len(val) > 5000:
                val = val[:5000]
            if total_chars + len(key) + len(val) > 50000:
                break
            raw_meta[key] = val
            total_chars += len(key) + len(val)

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
                skipped_nontext=0,
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
    skipped_nontext = 0

    async def worker(sr: SearchResult, client: httpx.AsyncClient):
        nonlocal skipped_robots, failed, below_threshold, fetched, skipped_nontext

        async with semaphore:
            allowed = await _respect_robots(
                sr.url,
                cfg.user_agent,
                timeout=cfg.robots_timeout_seconds,
            )
            if not allowed:
                skipped_robots += 1
                return
            resp = await _fetch_with_retries(client, sr.url, cfg)
            if resp is None or resp.content is None:
                failed += 1
                return

            fetched += 1
            content_type = resp.headers.get("Content-Type", "")
            if _is_nontext_content_type(content_type):
                skipped_nontext += 1
                return
            doc: Optional[ExtractedDocument]
            if _is_pdf(sr.url, content_type):
                doc = _extract_from_pdf(resp.content, sr.url, sr)
            else:
                try:
                    html_text = resp.text
                except UnicodeDecodeError:
                    html_text = resp.content.decode("utf-8", errors="ignore")
                doc = _extract_from_html(html_text, sr.url, sr)

                # Optional fallback for JS-heavy pages: if extraction failed, try a browser render.
                if not doc and cfg.enable_browser_fallback:
                    rendered = await _fetch_with_playwright(
                        sr.url,
                        user_agent=cfg.user_agent,
                        timeout_seconds=cfg.request_timeout_seconds,
                    )
                    if rendered:
                        doc = _extract_from_html(rendered, sr.url, sr)

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

    async with httpx.AsyncClient(
        headers={"User-Agent": cfg.user_agent},
        follow_redirects=True,
    ) as client:
        tasks = [worker(sr, client) for sr in search_results]
        # Run tasks with backpressure
        await asyncio.gather(*tasks)

    stats = Phase3Stats(
        total_input_urls=len(search_results),
        fetched=fetched,
        successful=len(documents),
        skipped_robots=skipped_robots,
        skipped_nontext=skipped_nontext,
        failed=failed,
        below_quality_threshold=below_threshold,
    )

    return Phase3Output(
        original_query=original_query,
        documents=documents,
        stats=stats,
        input_urls=search_results,
    )


"""
Phase 3 core: fetch HTML/PDF, extract main content, and clean it.

Implements:
- Task 3.1 Web Scraping (HTML + PDF, metadata, boilerplate removal)
- Task 3.2 Content Cleaning (whitespace, dedupe, quality filter)
- Task 3.3 Rate Limiting (max concurrent requests, simple backoff, robots.txt)
"""

import asyncio
import json
import re
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from hashlib import sha256
from typing import Optional
from urllib.parse import urlparse
from urllib import robotparser

import fitz  # pymupdf
import httpx
import trafilatura

from app.phases.phase2.schemas import SearchResult
from app.phases.phase3.schemas import ExtractedDocument, Phase3Output, Phase3Stats


@dataclass
class Phase3Config:
    max_concurrent_requests: int = 8
    max_requests_per_domain: int = 2  # max concurrent requests to same domain
    request_timeout_seconds: float = 15.0
    robots_timeout_seconds: float = 5.0
    max_retries: int = 3
    backoff_base_seconds: float = 0.5
    min_word_count: int = 200
    user_agent: str = "EmbeddingFlowBot/0.1 (+https://github.com/APotta1/EmbeddingFlow)"
    enable_browser_fallback: bool = False
    unpaywall_email: str = ""
    crossref_email: str = ""
    core_api_key: str = ""


@dataclass
class WorkerResult:
    doc: Optional[ExtractedDocument]
    status: str  # "success" | "failed" | "robots" | "nontext" | "below_threshold" | "duplicate"
    source_type: str = "http"  # "http" | "api" | "playwright" | "pdf"


_robots_cache: dict[str, tuple[robotparser.RobotFileParser, float]] = {}
ROBOTS_CACHE_TTL = 86400  # 24 hours
_robots_lock: Optional[asyncio.Lock] = None

_domain_semaphores: dict[str, asyncio.Semaphore] = {}
_domain_semaphores_lock = threading.Lock()


def _get_robots_lock() -> asyncio.Lock:
    global _robots_lock
    if _robots_lock is None:
        _robots_lock = asyncio.Lock()
    return _robots_lock


def _get_domain_semaphore(domain: str, max_per_domain: int) -> asyncio.Semaphore:
    with _domain_semaphores_lock:
        if domain not in _domain_semaphores:
            _domain_semaphores[domain] = asyncio.Semaphore(max_per_domain)
        return _domain_semaphores[domain]


async def _get_robots_parser(
    domain: str,
    scheme: str = "https",
    *,
    timeout: float = 5.0,
) -> robotparser.RobotFileParser:
    key = f"{scheme}://{domain}"
    async with _get_robots_lock():
        if key in _robots_cache:
            parser, expires_at = _robots_cache[key]
            if time.time() < expires_at:
                return parser

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

        _robots_cache[key] = (rp, time.time() + ROBOTS_CACHE_TTL)
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
    try:
        from playwright.async_api import async_playwright  # type: ignore[import-not-found]
        from playwright_stealth import Stealth  # type: ignore[import-not-found]
    except Exception:
        return None

    timeout_ms = int(max(1.0, timeout_seconds) * 1000)
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            try:
                page = await browser.new_page(user_agent=user_agent)
                async with Stealth().use_async(page):
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


def _is_academic_url(url: str) -> bool:
    """Detect academic/reference URLs that have different prose structure than general web content."""
    ACADEMIC_DOMAINS = {
        "pmc.ncbi.nlm.nih.gov",
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "arxiv.org",
        "frontiersin.org",
        "en.wikipedia.org",
        "journals.plos.org",
        "elifesciences.org",
        "biorxiv.org",
        "medrxiv.org",
        "jamanetwork.com",
        "bmj.com",
        "thelancet.com",
        "nejm.org",
        "academic.oup.com",
        "journals.sagepub.com",
        "psycnet.apa.org",
        "ieeexplore.ieee.org",
        "dl.acm.org",
        "semanticscholar.org",
        "scholar.google.com",
        "researchgate.net",
        "ssrn.com",
        "hal.science",
        "jneurosci.org",
        "cell.com",
        "science.org",
        "pnas.org",
    }
    try:
        domain = urlparse(url).netloc.lower()
        return any(domain == d or domain.endswith("." + d) for d in ACADEMIC_DOMAINS)
    except Exception:
        return False


def _passes_quality_filter(content: str, min_word_count: int, url: str = "") -> bool:
    """
    Quality filter using three signals:
    1. Raw word count minimum
    2. Average sentence length — real prose averages 10+ words per sentence
    3. Unique word ratio — boilerplate repeats tokens, real content has variety
    """
    words = content.split()
    if len(words) < min_word_count:
        return False

    # Average sentence length — split on sentence-ending punctuation
    sentences = [s.strip() for s in re.split(r"[.!?]+", content) if s.strip()]
    if sentences:
        avg_sentence_len = len(words) / len(sentences)
        min_avg_sentence_len = 5 if _is_academic_url(url) else 8
        if avg_sentence_len < min_avg_sentence_len:
            return False

    # Unique word ratio — boilerplate has low lexical diversity
    unique_ratio = len({w.lower() for w in words}) / len(words)
    if unique_ratio < 0.2:  # less than 20% unique words = repetitive/low quality
        return False

    return True


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


def _extract_from_pdf(
    content: bytes,
    url: str,
    search_result: SearchResult,
) -> Optional[ExtractedDocument]:
    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception:
        return None

    try:
        text_chunks: list[str] = []
        for page in doc:
            blocks = page.get_text("blocks")
            # block structure: (x0, y0, x1, y1, text, block_no, block_type)
            # block_type 0 = text, 1 = image — skip image blocks
            page_text = "\n".join(
                b[4].strip()
                for b in sorted(blocks, key=lambda b: (b[1], b[0]))
                if b[6] == 0 and b[4].strip()
            )
            if page_text:
                text_chunks.append(page_text)

        full_text = "\n\n".join(text_chunks).strip()
        if not full_text:
            return None

        normalized = _normalize_whitespace(full_text)
        paragraphs = _split_paragraphs(full_text)
        domain = urlparse(url).netloc or search_result.domain

        meta = doc.metadata or {}
        title = meta.get("title") or search_result.title
        author = meta.get("author")
        publish_date = meta.get("creationDate") or meta.get("modDate")

        return ExtractedDocument(
            url=url,
            title=title,
            author=author,
            publish_date=publish_date,
            domain=domain,
            source_api=search_result.source_api,
            position=search_result.position,
            content=normalized,
            content_paragraphs=paragraphs,
            raw_metadata={"pdf_metadata": {k: str(v) for k, v in meta.items() if v}},
        )
    except Exception:
        return None
    finally:
        doc.close()


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
            parsed = json.loads(extracted_json)
            data = parsed if isinstance(parsed, dict) else {}
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


_PMC_ID_RE = re.compile(r"PMC(\d+)", re.IGNORECASE)
_PUBMED_ID_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", re.IGNORECASE)
_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_EUTILS_PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _extract_pmc_id(url: str) -> Optional[str]:
    """Extract PMC ID from pmc.ncbi.nlm.nih.gov URL or PMID from pubmed.ncbi.nlm.nih.gov URL."""
    m = _PMC_ID_RE.search(url)
    if m:
        return m.group(1)
    return None


def _extract_pubmed_id(url: str) -> Optional[str]:
    """Extract PMID from a pubmed.ncbi.nlm.nih.gov URL."""
    m = _PUBMED_ID_RE.search(url)
    return m.group(1) if m else None


async def _fetch_pmc_article(
    pmc_id: str,
    client: httpx.AsyncClient,
    timeout: float = 15.0,
) -> Optional[str]:
    """
    Fetch full article text from NCBI E-utilities API.
    Returns plain text content or None on failure.
    """
    params = {
        "db": "pmc",
        "id": f"PMC{pmc_id}",
        "rettype": "xml",
        "retmode": "xml",
    }
    try:
        resp = await client.get(_EUTILS_BASE, params=params, timeout=timeout)
        if resp.status_code != 200 or not resp.text:
            return None

        # Parse XML and extract text from <body> paragraphs
        root = ET.fromstring(resp.text)

        # Collect all text nodes from <p> tags in article body
        paragraphs = []
        for elem in root.iter():
            if elem.tag in ("p", "title", "abstract"):
                text = "".join(elem.itertext()).strip()
                if text:
                    paragraphs.append(text)

        return "\n\n".join(paragraphs) if paragraphs else None
    except Exception:
        return None


async def _fetch_pubmed_abstract(
    pmid: str,
    client: httpx.AsyncClient,
    timeout: float = 15.0,
) -> Optional[str]:
    """
    Fetch abstract text from NCBI E-utilities API using PMID.
    Returns plain text or None on failure.
    """
    params = {
        "db": "pubmed",
        "id": pmid,
        "rettype": "abstract",
        "retmode": "xml",
    }
    try:
        resp = await client.get(_EUTILS_PUBMED_BASE, params=params, timeout=timeout)
        if resp.status_code != 200 or not resp.text:
            return None

        root = ET.fromstring(resp.text)
        sections: list[str] = []

        # Extract article title
        for elem in root.iter("ArticleTitle"):
            text = "".join(elem.itertext()).strip()
            if text:
                sections.append(text)

        # Extract abstract text — may have multiple AbstractText elements with labels
        for elem in root.iter("AbstractText"):
            label = elem.get("Label", "")
            text = "".join(elem.itertext()).strip()
            if text:
                sections.append(f"{label}: {text}" if label else text)

        return "\n\n".join(sections) if sections else None
    except Exception:
        return None


_WIKIPEDIA_API_BASE = "https://en.wikipedia.org/api/rest_v1/page"


def _extract_wikipedia_title(url: str) -> Optional[str]:
    try:
        parsed = urlparse(url)
        if "wikipedia.org" not in parsed.netloc:
            return None
        parts = parsed.path.split("/wiki/")
        if len(parts) < 2:
            return None
        return parts[1].split("#")[0]
    except Exception:
        return None


async def _fetch_wikipedia_article(
    title: str,
    client: httpx.AsyncClient,
    timeout: float = 15.0,
) -> Optional[str]:
    try:
        url = f"{_WIKIPEDIA_API_BASE}/html/{title}"
        resp = await client.get(url, timeout=timeout)
        if resp.status_code == 200 and resp.text:
            text = trafilatura.extract(resp.text, include_comments=False)
            if text and len(text.split()) > 100:
                return text
        # Fallback to summary
        url = f"{_WIKIPEDIA_API_BASE}/summary/{title}"
        resp = await client.get(url, timeout=timeout)
        if resp.status_code == 200:
            data = json.loads(resp.text)
            extract = data.get("extract", "")
            if extract:
                return extract
    except Exception:
        pass
    return None


def _extract_doi(url: str) -> Optional[str]:
    try:
        parsed = urlparse(url)
        if "doi.org" in parsed.netloc:
            return parsed.path.lstrip("/")
        m = re.search(r"(10\.\d{4,}/[^\s\"'<>]+)", url)
        return m.group(1) if m else None
    except Exception:
        return None


_UNPAYWALL_BASE = "https://api.unpaywall.org/v2"


async def _fetch_unpaywall(
    doi: str,
    email: str,
    client: httpx.AsyncClient,
    timeout: float = 15.0,
) -> Optional[str]:
    if not email:
        return None
    try:
        url = f"{_UNPAYWALL_BASE}/{doi}"
        resp = await client.get(url, params={"email": email}, timeout=timeout)
        if resp.status_code != 200:
            return None
        data = json.loads(resp.text)
        best_oa = data.get("best_oa_location")
        if best_oa:
            return best_oa.get("url_for_pdf") or best_oa.get("url")
    except Exception:
        pass
    return None


_CORE_BASE = "https://api.core.ac.uk/v3"


async def _fetch_core(
    doi: str,
    api_key: str,
    client: httpx.AsyncClient,
    timeout: float = 15.0,
) -> Optional[str]:
    if not api_key:
        return None
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        params = {"q": f"doi:{doi}", "limit": 1}
        resp = await client.get(
            f"{_CORE_BASE}/search/works",
            params=params,
            headers=headers,
            timeout=timeout,
        )
        if resp.status_code != 200:
            return None
        data = json.loads(resp.text)
        results = data.get("results", [])
        if not results:
            return None
        paper = results[0]
        full_text = paper.get("fullText")
        if full_text and len(full_text.split()) > 100:
            return full_text
        # Fallback to abstract
        abstract = paper.get("abstract", "")
        title = paper.get("title", "")
        if abstract:
            return f"{title}\n\n{abstract}" if title else abstract
    except Exception:
        pass
    return None


_CROSSREF_BASE = "https://api.crossref.org/works"


async def _fetch_crossref(
    doi: str,
    email: str,
    client: httpx.AsyncClient,
    timeout: float = 15.0,
) -> Optional[str]:
    try:
        headers = {
            "User-Agent": f"EmbeddingFlow/0.1 (mailto:{email or 'user@example.com'})"
        }
        url = f"{_CROSSREF_BASE}/{doi}"
        resp = await client.get(url, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            return None
        data = json.loads(resp.text)
        item = data.get("message", {})

        title_list = item.get("title", [])
        title = title_list[0] if title_list else ""

        abstract = item.get("abstract", "")
        abstract = re.sub(r"<[^>]+>", "", abstract).strip()

        authors = []
        for a in item.get("author", [])[:5]:
            name = f"{a.get('given', '')} {a.get('family', '')}".strip()
            if name:
                authors.append(name)
        author_str = ", ".join(authors)

        year = ""
        published = item.get("published", {}).get("date-parts", [[]])[0]
        if published:
            year = str(published[0])

        if title and abstract:
            return f"{title}\n\n{author_str} ({year})\n\n{abstract}"
        if title:
            return title
    except Exception:
        pass
    return None


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
                fetched_via_api=0,
                fetched_via_playwright=0,
            ),
            input_urls=[],
        )

    semaphore = asyncio.Semaphore(cfg.max_concurrent_requests)

    def _make_doc(
        text: str,
        url: str,
        sr: SearchResult,
        source: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        publish_date: Optional[str] = None,
    ) -> ExtractedDocument:
        normalized = _normalize_whitespace(text)
        paragraphs = _split_paragraphs(text)
        return ExtractedDocument(
            url=url,
            title=title or sr.title,
            author=author,
            publish_date=publish_date,
            domain=urlparse(url).netloc or sr.domain,
            source_api=sr.source_api,
            position=sr.position,
            content=normalized,
            content_paragraphs=paragraphs,
            raw_metadata={"source": source},
        )

    async def worker(sr: SearchResult, client: httpx.AsyncClient) -> WorkerResult:
        async with semaphore:
            # -- STEP 0: robots.txt check ----------------------------------------
            allowed = await _respect_robots(
                sr.url, cfg.user_agent, timeout=cfg.robots_timeout_seconds
            )
            if not allowed:
                return WorkerResult(doc=None, status="robots", source_type="http")

            # -- STEP 1: PMC full-text API ---------------------------------------
            if "pmc.ncbi.nlm.nih.gov" in sr.url:
                pmc_id = _extract_pmc_id(sr.url)
                if pmc_id:
                    text = await _fetch_pmc_article(pmc_id, client, timeout=cfg.request_timeout_seconds)
                    if text:
                        doc = _make_doc(text, sr.url, sr, source="ncbi_pmc_api")
                        if not _passes_quality_filter(doc.content, cfg.min_word_count, url=sr.url):
                            return WorkerResult(doc=None, status="below_threshold", source_type="api")
                        return WorkerResult(doc=doc, status="success", source_type="api")

            # -- STEP 2: PubMed abstract API -------------------------------------
            if "pubmed.ncbi.nlm.nih.gov" in sr.url:
                pubmed_id = _extract_pubmed_id(sr.url)
                if pubmed_id:
                    text = await _fetch_pubmed_abstract(pubmed_id, client, timeout=cfg.request_timeout_seconds)
                    if text:
                        doc = _make_doc(text, sr.url, sr, source="ncbi_pubmed_api")
                        if not _passes_quality_filter(doc.content, min_word_count=50, url=sr.url):
                            return WorkerResult(doc=None, status="below_threshold", source_type="api")
                        return WorkerResult(doc=doc, status="success", source_type="api")

            # -- STEP 3: Wikipedia REST API --------------------------------------
            if "wikipedia.org" in sr.url:
                wiki_title = _extract_wikipedia_title(sr.url)
                if wiki_title:
                    text = await _fetch_wikipedia_article(wiki_title, client, timeout=cfg.request_timeout_seconds)
                    if text:
                        doc = _make_doc(text, sr.url, sr, source="wikipedia_rest_api")
                        if not _passes_quality_filter(doc.content, cfg.min_word_count, url=sr.url):
                            return WorkerResult(doc=None, status="below_threshold", source_type="api")
                        return WorkerResult(doc=doc, status="success", source_type="api")

            # -- STEP 4: DOI -> Unpaywall -> CORE -> CrossRef --------------------
            doi = _extract_doi(sr.url)
            if doi:
                # 4a. Unpaywall — best case: legal free full PDF
                if cfg.unpaywall_email:
                    pdf_url = await _fetch_unpaywall(
                        doi, cfg.unpaywall_email, client,
                        timeout=cfg.request_timeout_seconds,
                    )
                    if pdf_url:
                        pdf_resp = await _fetch_with_retries(client, pdf_url, cfg)
                        if pdf_resp and _is_pdf(pdf_url, pdf_resp.headers.get("Content-Type", "")):
                            doc = _extract_from_pdf(pdf_resp.content, sr.url, sr)
                            if doc:
                                if not _passes_quality_filter(doc.content, cfg.min_word_count, url=sr.url):
                                    return WorkerResult(doc=None, status="below_threshold", source_type="pdf")
                                return WorkerResult(doc=doc, status="success", source_type="pdf")

                # 4b. CORE — full text for open access papers
                if cfg.core_api_key:
                    core_text = await _fetch_core(
                        doi, cfg.core_api_key, client,
                        timeout=cfg.request_timeout_seconds,
                    )
                    if core_text:
                        doc = _make_doc(core_text, sr.url, sr, source="core_api")
                        if not _passes_quality_filter(doc.content, cfg.min_word_count, url=sr.url):
                            return WorkerResult(doc=None, status="below_threshold", source_type="api")
                        return WorkerResult(doc=doc, status="success", source_type="api")

                # 4c. CrossRef — metadata + abstract as last resort
                crossref_text = await _fetch_crossref(
                    doi,
                    email=cfg.crossref_email or cfg.unpaywall_email,
                    client=client,
                    timeout=cfg.request_timeout_seconds,
                )
                if crossref_text:
                    doc = _make_doc(crossref_text, sr.url, sr, source="crossref_api")
                    if not _passes_quality_filter(doc.content, min_word_count=50, url=sr.url):
                        return WorkerResult(doc=None, status="below_threshold", source_type="api")
                    return WorkerResult(doc=doc, status="success", source_type="api")

            # -- STEP 5: Normal httpx fetch --------------------------------------
            resp = await _fetch_with_retries(client, sr.url, cfg)

            # -- STEP 6: Playwright stealth fallback on 403 ----------------------
            if resp is None and cfg.enable_browser_fallback:
                rendered = await _fetch_with_playwright(
                    sr.url,
                    user_agent=cfg.user_agent,
                    timeout_seconds=cfg.request_timeout_seconds,
                )
                if rendered:
                    doc = _extract_from_html(rendered, sr.url, sr)
                    if doc:
                        if not _passes_quality_filter(doc.content, cfg.min_word_count, url=sr.url):
                            return WorkerResult(doc=None, status="below_threshold", source_type="playwright")
                        return WorkerResult(doc=doc, status="success", source_type="playwright")
                return WorkerResult(doc=None, status="failed", source_type="playwright")

            if resp is None or resp.content is None:
                return WorkerResult(doc=None, status="failed", source_type="http")

            # -- STEP 7: Extract content from response ---------------------------
            content_type = resp.headers.get("Content-Type", "")
            if _is_nontext_content_type(content_type):
                return WorkerResult(doc=None, status="nontext", source_type="http")

            if _is_pdf(sr.url, content_type):
                doc = _extract_from_pdf(resp.content, sr.url, sr)
                source_type = "pdf"
            else:
                try:
                    html_text = resp.text
                except UnicodeDecodeError:
                    html_text = resp.content.decode("utf-8", errors="ignore")
                doc = _extract_from_html(html_text, sr.url, sr)
                source_type = "http"

            if not doc:
                return WorkerResult(doc=None, status="failed", source_type=source_type)

            if not _passes_quality_filter(doc.content, cfg.min_word_count, url=sr.url):
                return WorkerResult(doc=None, status="below_threshold", source_type=source_type)

            return WorkerResult(doc=doc, status="success", source_type=source_type)

    async with httpx.AsyncClient(
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
        },
        follow_redirects=True,
    ) as client:
        results: list[WorkerResult] = await asyncio.gather(*[worker(sr, client) for sr in search_results])
    fetched_via_api = sum(1 for r in results if r.status == "success" and r.source_type == "api")
    fetched_via_playwright = sum(
        1 for r in results if r.status == "success" and r.source_type == "playwright"
    )

    seen_hashes: set[str] = set()
    documents: list[ExtractedDocument] = []
    skipped_robots = failed = below_threshold = fetched = skipped_nontext = 0

    for r in results:
        if r.status == "robots":
            skipped_robots += 1
        elif r.status == "nontext":
            skipped_nontext += 1
            fetched += 1
        elif r.status == "below_threshold":
            below_threshold += 1
            fetched += 1
        elif r.status == "failed":
            failed += 1
        elif r.status == "success" and r.doc:
            fetched += 1
            h = _hash_content(r.doc.content)
            if h not in seen_hashes:
                seen_hashes.add(h)
                documents.append(r.doc)

    stats = Phase3Stats(
        total_input_urls=len(search_results),
        fetched=fetched,
        successful=len(documents),
        skipped_robots=skipped_robots,
        skipped_nontext=skipped_nontext,
        failed=failed,
        below_quality_threshold=below_threshold,
        fetched_via_api=fetched_via_api,
        fetched_via_playwright=fetched_via_playwright,
    )

    return Phase3Output(
        original_query=original_query,
        documents=documents,
        stats=stats,
        input_urls=search_results,
    )


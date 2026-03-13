
from __future__ import annotations

import re
from typing import List, Optional

from app.phases.phase3.schemas import ExtractedDocument
from app.phases.phase4.schemas import Chunk, Phase4Config  # type: ignore[import-not-found]

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"\'])")


def _split_sentences(text: str) -> List[str]:
    """
    Best-effort sentence splitter (no external deps).

    This is intentionally conservative: it splits on punctuation + whitespace
    when the next token looks like a sentence start. If it can't split well,
    it returns the original text as a single segment.
    """

    t = (text or "").strip()
    if not t:
        return []
    parts = _SENTENCE_SPLIT_RE.split(t)
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts if len(parts) > 1 else [t]


def _split_by_token_windows(text: str, max_tokens: int, *, overlap_tokens: int = 0) -> List[str]:
    """
    Hard fallback splitter for very long text: split into word windows.

    Keeps approximate overlap between windows to preserve continuity.
    """

    words = [w for w in (text or "").split() if w]
    if not words:
        return []
    if max_tokens <= 0:
        return [" ".join(words)]

    out: List[str] = []
    step = max(1, max_tokens - max(0, overlap_tokens))
    i = 0
    while i < len(words):
        window = words[i : i + max_tokens]
        out.append(" ".join(window).strip())
        if i + max_tokens >= len(words):
            break
        i += step
    return out


def _expand_oversized_paragraphs(
    paragraphs: List[str],
    *,
    max_tokens: int,
    window_overlap_tokens: int,
) -> tuple[List[str], List[int]]:
    """
    Paragraph-first segmentation with sentence/window fallback.

    Returns:
      - units: list of text units (paragraphs and/or sentence/window segments)
      - unit_to_paragraph_index: mapping each unit back to its original paragraph index
    """

    units: List[str] = []
    mapping: List[int] = []

    for p_idx, p in enumerate(paragraphs):
        p = (p or "").strip()
        if not p:
            continue

        if _approx_token_count(p) <= max_tokens:
            units.append(p)
            mapping.append(p_idx)
            continue

        # Oversized paragraph: split into sentences
        sentences = _split_sentences(p)

        # If sentence splitting didn't help (still huge or 1 segment), fall back to token windows
        if len(sentences) <= 1 and _approx_token_count(sentences[0]) > max_tokens:
            windows = _split_by_token_windows(
                sentences[0],
                max_tokens=max_tokens,
                overlap_tokens=window_overlap_tokens,
            )
            for w in windows:
                units.append(w)
                mapping.append(p_idx)
            continue

        # Pack sentences into sub-units up to max_tokens
        buf: List[str] = []
        buf_tokens = 0
        for s in sentences:
            s_tokens = _approx_token_count(s)
            if buf and buf_tokens + s_tokens > max_tokens:
                units.append(" ".join(buf).strip())
                mapping.append(p_idx)
                buf = []
                buf_tokens = 0

            buf.append(s)
            buf_tokens += s_tokens

        if buf:
            units.append(" ".join(buf).strip())
            mapping.append(p_idx)

    return units, mapping


def _approx_token_count(text: str) -> int:
    """
    Rough token estimator.

    For now we approximate tokens as number of whitespace-delimited words,
    which is usually within a small factor of true token count for English.
    This can be swapped out for a real tokenizer later.
    """

    # Split on whitespace; ignore very short fragments
    return sum(1 for w in text.split() if w)


def _paragraph_similarity_jaccard(a: str, b: str) -> float:
    """
    Very lightweight lexical similarity between two paragraphs.

    This is not a full ML model; it's a cheap proxy that still helps us
    identify sharp topic shifts. We compute Jaccard similarity over
    lowercased word sets with simple stopword removal.
    """

    # Basic tokenization
    tokenize = lambda s: set(
        w
        for w in re.findall(r"\b\w+\b", s.lower())
        if len(w) > 2 and w not in _STOPWORDS
    )
    sa = tokenize(a)
    sb = tokenize(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    if union == 0:
        return 0.0
    return inter / union


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "are",
    "was",
    "were",
    "will",
    "would",
    "could",
    "should",
    "about",
    "into",
    "over",
    "such",
    "their",
    "there",
    "have",
    "has",
    "had",
    "not",
    "but",
    "than",
}


def _adjacent_similarities_tfidf(paragraphs: List[str]) -> List[float]:
    """
    Compute cosine similarity between adjacent paragraphs using TF-IDF vectors.

    Returns a list of length len(paragraphs) - 1 where sims[i] compares
    paragraphs[i] vs paragraphs[i+1].
    """

    # Local import so Phase 4 can still run without sklearn if configured otherwise.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if len(paragraphs) < 2:
        return []

    vec = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
        lowercase=True,
    )
    X = vec.fit_transform(paragraphs)

    sims: List[float] = []
    for i in range(X.shape[0] - 1):
        s = float(cosine_similarity(X[i], X[i + 1])[0][0])
        if s != s:  # NaN
            s = 0.0
        sims.append(max(0.0, min(1.0, s)))
    return sims


def _adjacent_similarities_jaccard(paragraphs: List[str]) -> List[float]:
    if len(paragraphs) < 2:
        return []
    return [
        _paragraph_similarity_jaccard(paragraphs[i], paragraphs[i + 1])
        for i in range(len(paragraphs) - 1)
    ]


def _compute_adjacent_similarities(
    paragraphs: List[str], config: Phase4Config
) -> Optional[List[float]]:
    """
    Compute adjacent paragraph similarities per the configured boundary model.

    Returns:
      - list[float] of length len(paragraphs)-1 when available
      - None if computation is unavailable (should be rare; we fall back to jaccard)
    """

    model = (getattr(config, "boundary_model", "jaccard") or "jaccard").strip().lower()
    if model == "tfidf":
        try:
            return _adjacent_similarities_tfidf(paragraphs)
        except Exception:
            return _adjacent_similarities_jaccard(paragraphs)
    return _adjacent_similarities_jaccard(paragraphs)


def _ensure_paragraphs(doc: ExtractedDocument) -> List[str]:
    """
    Get a robust list of paragraphs for a document.

    Preference:
    - If content_paragraphs is populated, use it.
    - Otherwise, split content on double newlines.
    """

    if doc.content_paragraphs:
        return list(doc.content_paragraphs)
    # Fallback: simple double-newline split
    paragraphs = re.split(r"\n{2,}", doc.content)
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_document(
    doc: ExtractedDocument,
    document_index: int,
    config: Phase4Config,
) -> List[Chunk]:
    """
    Chunk a single ExtractedDocument into overlapping, paragraph-aware chunks.

    The algorithm:
    - Start from the first paragraph.
    - Add paragraphs until we reach max_chunk_tokens, but try not to cross
      very low-similarity paragraph boundaries unless needed to satisfy
      min_chunk_tokens.
    - Emit a chunk; then start the next chunk with an overlap window.
    """

    paragraphs = _ensure_paragraphs(doc)
    if not paragraphs:
        return []

    min_tokens = config.min_chunk_tokens
    max_tokens = max(config.max_chunk_tokens, min_tokens + 50)
    overlap_tokens = min(config.overlap_tokens, min_tokens)

    chunks: List[Chunk] = []

    # Paragraph-first, but split oversized paragraphs into sentence/window units so we can
    # obey max_tokens without producing giant single-paragraph chunks.
    units, unit_to_para = _expand_oversized_paragraphs(
        paragraphs,
        max_tokens=max_tokens,
        window_overlap_tokens=max(25, overlap_tokens // 4),
    )
    if not units:
        return []

    unit_tokens = [_approx_token_count(u) for u in units]
    adjacent_sims = _compute_adjacent_similarities(units, config)
    n = len(units)
    start_u = 0

    while start_u < n:
        # Determine end index for this chunk
        current_tokens = 0
        end_u = start_u

        while end_u < n:
            next_tokens = unit_tokens[end_u]
            if current_tokens + next_tokens > max_tokens:
                # Stop if we've reached the max size
                break

            # If this paragraph is a hard topic shift and we already have
            # enough tokens, prefer to break here.
            if (
                end_u > start_u
                and current_tokens >= min_tokens
                and (
                    (adjacent_sims is not None and adjacent_sims[end_u - 1] < config.min_paragraph_similarity)
                    or (
                        adjacent_sims is None
                        and _paragraph_similarity_jaccard(units[end_u - 1], units[end_u])
                        < config.min_paragraph_similarity
                    )
                )
            ):
                break

            current_tokens += next_tokens
            end_u += 1

        # Ensure we always make progress
        if end_u == start_u:
            end_u = min(start_u + 1, n)
            current_tokens = unit_tokens[start_u]

        # Build chunk text
        span_units = units[start_u:end_u]
        raw_text = "\n\n".join(span_units).strip()
        approx_tokens = _approx_token_count(raw_text)

        chunk_index = len(chunks)
        start_para = unit_to_para[start_u]
        end_para = unit_to_para[end_u - 1] + 1
        chunks.append(
            Chunk(
                document_index=document_index,
                chunk_index=chunk_index,
                url=doc.url,
                title=doc.title,
                domain=doc.domain,
                source_api=doc.source_api,
                publish_date=doc.publish_date,
                start_paragraph_index=start_para,
                end_paragraph_index=end_para,
                raw_text=raw_text,
                contextualized_text=raw_text,  # will be filled/updated by contextualizer
                approx_token_count=approx_tokens,
            )
        )

        # Move start index forward with overlap.
        if end_u >= n:
            break

        # Compute how many tokens to step forward to maintain overlap.
        # We want the next chunk to start such that we keep roughly overlap_tokens
        # from the end of the current chunk.
        remaining = current_tokens
        step_u = end_u
        while step_u > start_u and remaining > overlap_tokens:
            step_u -= 1
            remaining -= unit_tokens[step_u]

        # Ensure we still move forward
        start_u = max(step_u, start_u + 1)

    return chunks


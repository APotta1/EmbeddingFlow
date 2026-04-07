from __future__ import annotations

import os
from typing import Any

import httpx

from app.phases.phase5.schemas import EmbeddingProvider


def _chunks(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _openai_embed(texts: list[str], model: str) -> list[list[float]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()
    return [item["embedding"] for item in data.get("data", [])]


def _voyage_embed(texts: list[str], model: str) -> list[list[float]]:
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError("VOYAGE_API_KEY is required for Voyage embeddings.")

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()
    return [item["embedding"] for item in data.get("data", [])]


def _cohere_embed(texts: list[str], model: str) -> list[list[float]]:
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY is required for Cohere embeddings.")

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            "https://api.cohere.ai/v1/embed",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "texts": texts,
                "input_type": "search_document",
                "embedding_types": ["float"],
            },
        )
        resp.raise_for_status()
        data = resp.json()

    embeddings: list[list[float]] = []
    emb = data.get("embeddings", {})
    if isinstance(emb, dict) and "float" in emb:
        embeddings = emb["float"]
    elif isinstance(emb, list):
        embeddings = emb
    return embeddings


def embed_texts(
    texts: list[str],
    provider: EmbeddingProvider,
    model: str,
    batch_size: int,
) -> tuple[list[list[float]], list[dict[str, Any]]]:
    """
    Batch embedding generation for efficiency.

    Returns:
    - embeddings in the same order as `texts`
    - batch diagnostics (index, requested count, returned count)
    """
    if not texts:
        return [], []

    all_vectors: list[list[float]] = []
    diagnostics: list[dict[str, Any]] = []
    text_batches = _chunks(texts, batch_size)

    for batch_idx, batch in enumerate(text_batches):
        if provider == "openai":
            vectors = _openai_embed(batch, model=model)
        elif provider == "voyage":
            vectors = _voyage_embed(batch, model=model)
        elif provider == "cohere":
            vectors = _cohere_embed(batch, model=model)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

        diagnostics.append(
            {
                "batch_index": batch_idx,
                "requested": len(batch),
                "returned": len(vectors),
            }
        )
        all_vectors.extend(vectors)

    return all_vectors, diagnostics

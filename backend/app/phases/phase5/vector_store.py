from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import httpx

from app.phases.phase5.schemas import EmbeddingRecord, Phase5Config


class BaseVectorStore(ABC):
    @abstractmethod
    def upsert(self, records: list[EmbeddingRecord]) -> dict[str, Any]:
        raise NotImplementedError


class FaissVectorStore(BaseVectorStore):
    """
    Local vector store for demos.

    Saves:
    - FAISS index
    - ID mapping
    - metadata JSON
    """

    def __init__(self, config: Phase5Config):
        self._config = config

    def upsert(self, records: list[EmbeddingRecord]) -> dict[str, Any]:
        try:
            import faiss  # type: ignore[import-not-found]
            import numpy as np
        except Exception as exc:
            raise ValueError(
                "FAISS storage requires 'faiss-cpu' and 'numpy' installed."
            ) from exc

        if not records:
            return {"stored": 0, "index_path": None}

        dim = len(records[0].embedding)
        vectors = np.array([r.embedding for r in records], dtype="float32")

        if self._config.faiss_metric == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            vectors = vectors / norms
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatL2(dim)

        index.add(vectors)

        out_dir = "storage/faiss"
        index_path = f"{out_dir}/{self._config.collection_name}.index"
        ids_path = f"{out_dir}/{self._config.collection_name}.ids.json"
        metadata_path = f"{out_dir}/{self._config.collection_name}.metadata.json"

        import os

        os.makedirs(out_dir, exist_ok=True)
        faiss.write_index(index, index_path)
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump([r.id for r in records], f)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump([r.metadata.model_dump() for r in records], f)

        return {
            "stored": len(records),
            "index_path": index_path,
            "ids_path": ids_path,
            "metadata_path": metadata_path,
            "metric": self._config.faiss_metric,
        }


class QdrantVectorStore(BaseVectorStore):
    def __init__(self, config: Phase5Config):
        if not config.qdrant_url:
            raise ValueError("qdrant_url is required when vector_store='qdrant'.")
        self._config = config

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._config.qdrant_api_key:
            headers["api-key"] = self._config.qdrant_api_key
        return headers

    def _ensure_collection(self, dim: int) -> None:
        with httpx.Client(timeout=30.0) as client:
            resp = client.put(
                f"{self._config.qdrant_url}/collections/{self._config.collection_name}",
                headers=self._headers(),
                json={
                    "vectors": {
                        "size": dim,
                        "distance": "Cosine",
                    }
                },
            )
            # Qdrant returns 200 on create/update; any non-error is acceptable.
            resp.raise_for_status()

    def upsert(self, records: list[EmbeddingRecord]) -> dict[str, Any]:
        if not records:
            return {"stored": 0}

        self._ensure_collection(len(records[0].embedding))
        points = []
        for r in records:
            payload = r.metadata.model_dump()
            payload["text"] = r.text
            points.append({"id": r.id, "vector": r.embedding, "payload": payload})

        with httpx.Client(timeout=60.0) as client:
            resp = client.put(
                f"{self._config.qdrant_url}/collections/{self._config.collection_name}/points",
                headers=self._headers(),
                json={"points": points},
            )
            resp.raise_for_status()
            data = resp.json()

        return {"stored": len(records), "response": data}


class PineconeVectorStore(BaseVectorStore):
    def __init__(self, config: Phase5Config):
        if not config.pinecone_api_key:
            raise ValueError(
                "pinecone_api_key is required when vector_store='pinecone'."
            )
        if not config.pinecone_host and not config.pinecone_index_name:
            raise ValueError(
                "Either pinecone_host or pinecone_index_name is required for Pinecone."
            )
        self._config = config

    def _upsert_url(self) -> str:
        if self._config.pinecone_host:
            return f"https://{self._config.pinecone_host}/vectors/upsert"
        return f"https://{self._config.pinecone_index_name}.svc.pinecone.io/vectors/upsert"

    def upsert(self, records: list[EmbeddingRecord]) -> dict[str, Any]:
        if not records:
            return {"stored": 0}

        vectors = []
        for r in records:
            metadata = r.metadata.model_dump()
            metadata["text"] = r.text
            vectors.append(
                {
                    "id": r.id,
                    "values": r.embedding,
                    "metadata": metadata,
                }
            )

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                self._upsert_url(),
                headers={
                    "Api-Key": self._config.pinecone_api_key,
                    "Content-Type": "application/json",
                },
                json={"vectors": vectors},
            )
            resp.raise_for_status()
            data = resp.json()

        return {"stored": len(records), "response": data}


def get_vector_store(config: Phase5Config) -> BaseVectorStore:
    if config.vector_store == "faiss":
        return FaissVectorStore(config)
    if config.vector_store == "qdrant":
        return QdrantVectorStore(config)
    if config.vector_store == "pinecone":
        return PineconeVectorStore(config)
    raise ValueError(f"Unsupported vector store: {config.vector_store}")

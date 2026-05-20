"""
ChromaDB vector store wrapper for RAG.

This module hides direct ChromaDB calls behind a small stable interface.
It is used by:
- scripts/build_vector_db.py
- src/rag/retriever.py
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import chromadb
from chromadb.api.types import Embeddings, Metadatas

from src.rag.document import EmbeddingDocument

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChromaConfig:
    persist_dir: Path
    collection_name: str = "cultural_knowledge"


@dataclass(frozen=True)
class VectorSearchResult:
    doc_id: str
    text: str
    metadata: dict[str, Any]
    distance: float | None
    score: float | None


class ChromaVectorStore:
    """
    Thin wrapper around a persistent ChromaDB collection.
    """

    def __init__(self, config: ChromaConfig):
        self.config = config
        self.config.persist_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Opening ChromaDB persistent client at %s",
            self.config.persist_dir,
        )

        self.client = chromadb.PersistentClient(
            path=str(self.config.persist_dir),
        )

        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={
                "description": "Vietnamese cultural knowledge base",
                "hnsw:space": "cosine",
            },
        )

        logger.info(
            "Using Chroma collection: %s",
            self.config.collection_name,
        )

    @classmethod
    def from_config_dict(cls, config: dict[str, Any]) -> "ChromaVectorStore":
        vector_cfg = config.get("vector_db", {})

        persist_dir = vector_cfg.get("persist_dir")
        collection_name = vector_cfg.get("collection_name", "cultural_knowledge")

        if not persist_dir:
            raise ValueError("Missing vector_db.persist_dir in config.")

        return cls(
            ChromaConfig(
                persist_dir=Path(str(persist_dir)),
                collection_name=str(collection_name),
            )
        )

    def count(self) -> int:
        """Return number of documents in the collection."""
        return int(self.collection.count())

    def reset_collection(self) -> None:
        """
        Delete and recreate the collection.

        Use this for --recreate mode before full rebuild.
        """
        logger.warning(
            "Resetting Chroma collection: %s",
            self.config.collection_name,
        )

        try:
            self.client.delete_collection(self.config.collection_name)
        except Exception as exc:
            logger.info(
                "Collection delete skipped or failed harmlessly: %s",
                exc,
            )

        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={
                "description": "Vietnamese cultural knowledge base",
                "hnsw:space": "cosine",
            },
        )

    def upsert_documents(
        self,
        documents: list[EmbeddingDocument],
        embeddings: list[list[float]],
        *,
        batch_size: int = 512,
    ) -> int:
        """
        Upsert documents with precomputed embeddings.

        Args:
            documents: normalized EmbeddingDocument records.
            embeddings: same-length list of embedding vectors.
            batch_size: Chroma upsert batch size.

        Returns:
            Number of upserted documents.
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"documents and embeddings length mismatch: "
                f"{len(documents)} != {len(embeddings)}"
            )

        total = len(documents)

        if total == 0:
            return 0

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_docs = documents[start:end]
            batch_embeddings = embeddings[start:end]

            ids = [doc.doc_id for doc in batch_docs]
            texts = [doc.text for doc in batch_docs]
            metadatas: Metadatas = [doc.metadata for doc in batch_docs]
            chroma_embeddings = cast(Embeddings, batch_embeddings)

            self.collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=chroma_embeddings,
            )

            logger.info("Upserted Chroma docs: %d/%d", end, total)

        return total

    def query(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 3,
        where: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """
        Query collection by embedding vector.

        Chroma returns distances. For normalized embeddings, distance usually
        behaves like cosine distance depending on collection settings.
        We expose both distance and a simple score = 1 - distance when possible.
        """
        if not query_embedding:
            raise ValueError("query_embedding cannot be empty.")

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        results: list[VectorSearchResult] = []

        for doc_id, text, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            distance_value = float(distance) if distance is not None else None
            score = 1.0 - distance_value if distance_value is not None else None

            results.append(
                VectorSearchResult(
                    doc_id=str(doc_id),
                    text=str(text or ""),
                    metadata=dict(metadata or {}),
                    distance=distance_value,
                    score=score,
                )
            )

        return results

    def peek(self, limit: int = 5) -> dict[str, Any]:
        """Return a small collection sample for debugging."""
        return dict(self.collection.peek(limit=limit))


def delete_chroma_persist_dir(persist_dir: Path) -> None:
    """
    Hard-delete Chroma persistence directory.

    Usually reset_collection() is enough. This is only for local cleanup.
    """
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
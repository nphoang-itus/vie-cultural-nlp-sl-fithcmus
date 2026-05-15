"""
Retriever API for RAG.

This module combines:
- EmbeddingModel for query embedding
- ChromaVectorStore for vector search

It is the main interface that the inference pipeline should call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.rag.embedding import EmbeddingModel, build_embedding_model_from_config
from src.rag.vector_store import ChromaVectorStore, VectorSearchResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedContext:
    doc_id: str
    content: str
    metadata: dict[str, Any]
    distance: float | None
    score: float | None


class CulturalKnowledgeRetriever:
    """
    High-level retriever for Vietnamese cultural knowledge.
    """

    def __init__(
        self,
        embedder: EmbeddingModel,
        vector_store: ChromaVectorStore,
        default_top_k: int = 3,
        score_threshold: float | None = None,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.default_top_k = default_top_k
        self.score_threshold = score_threshold

    @classmethod
    def from_config_dict(cls, config: dict[str, Any]) -> "CulturalKnowledgeRetriever":
        retrieval_cfg = config.get("retrieval", {})

        embedder = build_embedding_model_from_config(config)
        vector_store = ChromaVectorStore.from_config_dict(config)

        return cls(
            embedder=embedder,
            vector_store=vector_store,
            default_top_k=int(retrieval_cfg.get("top_k", 3)),
            score_threshold=retrieval_cfg.get("score_threshold"),
        )

    @classmethod
    def from_config_file(
        cls,
        config_path: Path = Path("configs/rag.yaml"),
    ) -> "CulturalKnowledgeRetriever":
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        return cls.from_config_dict(config)

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedContext]:
        """
        Search cultural knowledge by natural language query.

        Args:
            query: User query or standalone question.
            top_k: Number of contexts to retrieve.
            filters: Optional Chroma metadata filter.
                Example:
                    {"category": "am_thuc"}
                    {"keyword": "bánh chưng"}
            score_threshold: Optional minimum score.
                If None, use instance-level score_threshold.

        Returns:
            List of retrieved contexts sorted by vector store ranking.
        """
        query = str(query or "").strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        resolved_top_k = top_k or self.default_top_k
        resolved_threshold = (
            self.score_threshold if score_threshold is None else score_threshold
        )

        query_embedding = self.embedder.embed_query(query)

        raw_results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=resolved_top_k,
            where=filters,
        )

        contexts = [
            self._to_retrieved_context(item)
            for item in raw_results
        ]

        if resolved_threshold is not None:
            contexts = [
                item for item in contexts
                if item.score is not None and item.score >= resolved_threshold
            ]

        return contexts

    def search_texts(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[str]:
        """
        Convenience method: return only retrieved contents.
        """
        return [
            item.content
            for item in self.search(query, top_k=top_k, filters=filters)
        ]

    def build_context_block(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        max_chars: int | None = None,
    ) -> str:
        """
        Retrieve contexts and concatenate them into one prompt-ready block.
        """
        contexts = self.search(query, top_k=top_k, filters=filters)

        parts: list[str] = []

        for idx, item in enumerate(contexts, start=1):
            keyword = item.metadata.get("keyword", "")
            category = item.metadata.get("category", "")

            part = (
                f"[Context {idx}]\n"
                f"doc_id: {item.doc_id}\n"
                f"keyword: {keyword}\n"
                f"category: {category}\n"
                f"content:\n{item.content}"
            )

            parts.append(part)

        block = "\n\n".join(parts)

        if max_chars is not None and len(block) > max_chars:
            block = block[:max_chars].rstrip()

        return block

    @staticmethod
    def _to_retrieved_context(result: VectorSearchResult) -> RetrievedContext:
        return RetrievedContext(
            doc_id=result.doc_id,
            content=result.text,
            metadata=result.metadata,
            distance=result.distance,
            score=result.score,
        )
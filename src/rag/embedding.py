"""
Embedding utilities for RAG.

This module wraps SentenceTransformer so the rest of the system does not depend
directly on sentence-transformers APIs.

Important contract:
- The same embedding model must be used for both ingestion and retrieval.
- Document embeddings and query embeddings should use the same normalization setting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def resolve_device(device: str = "auto") -> str:
    """
    Resolve embedding device.

    Supported:
    - auto: cuda > mps > cpu
    - cpu
    - cuda
    - mps
    """
    device = (device or "auto").lower()

    if device != "auto":
        return device

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str
    batch_size: int = 64
    normalize_embeddings: bool = True
    device: str = "auto"


class EmbeddingModel:
    """
    Thin wrapper around SentenceTransformer.
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.device = resolve_device(config.device)

        logger.info(
            "Loading embedding model: %s on device=%s",
            config.model_name,
            self.device,
        )

        self.model = SentenceTransformer(
            config.model_name,
            device=self.device,
        )

        self.embedding_dimension = self.model.get_embedding_dimension()

        logger.info(
            "Loaded embedding model: %s | dimension=%s",
            config.model_name,
            self.embedding_dimension,
        )

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Embed a list of texts.

        Returns:
            list[list[float]] with shape [len(texts), embedding_dimension]
        """
        cleaned_texts = [str(text or "").strip() for text in texts]

        if not cleaned_texts:
            return []

        if any(not text for text in cleaned_texts):
            raise ValueError("Cannot embed empty text.")

        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=len(cleaned_texts) >= 128,
        )

        return embeddings.astype("float32").tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Embed one query string.
        """
        query = str(query or "").strip()

        if not query:
            raise ValueError("Cannot embed empty query.")

        return self.embed_texts([query])[0]


def build_embedding_model_from_config(config: dict) -> EmbeddingModel:
    """
    Build EmbeddingModel from rag.yaml-like config dict.

    Expected:
        config["embedding"]["model_name"]
        config["embedding"]["batch_size"]
        config["embedding"]["normalize_embeddings"]
        config["embedding"]["device"]
    """
    emb_cfg = config.get("embedding", {})

    model_name = emb_cfg.get("model_name")
    if not model_name:
        raise ValueError("Missing embedding.model_name in config.")

    return EmbeddingModel(
        EmbeddingConfig(
            model_name=str(model_name),
            batch_size=int(emb_cfg.get("batch_size", 64)),
            normalize_embeddings=bool(emb_cfg.get("normalize_embeddings", True)),
            device=str(emb_cfg.get("device", "auto")),
        )
    )
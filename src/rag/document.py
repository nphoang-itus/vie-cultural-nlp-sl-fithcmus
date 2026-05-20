"""
Document normalization utilities for RAG ingestion.

This module converts validated KnowledgeDocument records into
EmbeddingDocument objects that are ready for embedding/vector storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chromadb.api.types import Metadata

from src.rag.knowledge_loader import KnowledgeDocument


@dataclass(frozen=True)
class EmbeddingDocument:
    """
    Internal document format used by the embedding/vector-store pipeline.
    """

    doc_id: str
    text: str
    metadata: Metadata


def count_words(text: str) -> int:
    """Count words using simple whitespace splitting."""
    text = (text or "").strip()
    if not text:
        return 0
    return len(text.split())


def normalize_metadata_for_chroma(metadata: dict[str, Any]) -> Metadata:
    """
    Normalize metadata into Chroma-compatible scalar values.

    Chroma metadata values should be simple scalar types:
    str, int, float, bool, or None.

    Lists such as source_image_ids are converted into a stable string.
    """
    normalized: Metadata = {}

    for key, value in metadata.items():
        if isinstance(value, list):
            normalized[key] = "||".join(str(item) for item in value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            normalized[key] = value
        else:
            normalized[key] = str(value)

    return normalized


def to_embedding_document(doc: KnowledgeDocument) -> EmbeddingDocument:
    """
    Convert one validated KnowledgeDocument into an EmbeddingDocument.

    The text is the exact content that will be embedded.
    Metadata is enriched with lightweight stats for debugging.
    """
    doc_id = str(doc["doc_id"]).strip()
    text = str(doc["content"]).strip()
    metadata = dict(doc.get("metadata", {}))

    enriched_metadata = {
        **metadata,
        "schema_version": doc.get("schema_version", "1.0"),
        "doc_id": doc_id,
        "content_length_chars": len(text),
        "content_length_words": count_words(text),
    }

    chroma_metadata = normalize_metadata_for_chroma(enriched_metadata)

    return EmbeddingDocument(
        doc_id=doc_id,
        text=text,
        metadata=chroma_metadata,
    )


def to_embedding_documents(
    docs: list[KnowledgeDocument],
) -> list[EmbeddingDocument]:
    """
    Convert many validated KnowledgeDocument records.
    """
    return [to_embedding_document(doc) for doc in docs]
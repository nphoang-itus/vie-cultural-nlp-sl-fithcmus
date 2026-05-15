"""
Build prompt-ready RAG context blocks from retrieved documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.rag.retriever import CulturalKnowledgeRetriever, RetrievedContext


@dataclass(frozen=True)
class RagContextResult:
    query: str
    contexts: list[RetrievedContext]
    context_block: str


def format_retrieved_context(
    context: RetrievedContext,
    *,
    index: int,
    include_metadata: bool = True,
) -> str:
    keyword = context.metadata.get("keyword", "")
    category = context.metadata.get("category", "")
    subcategory = context.metadata.get("subcategory", "")

    if include_metadata:
        return (
            f"[Cultural Context {index}]\n"
            f"doc_id: {context.doc_id}\n"
            f"keyword: {keyword}\n"
            f"category: {category}\n"
            f"subcategory: {subcategory}\n"
            f"content:\n{context.content}"
        )

    return (
        f"[Cultural Context {index}]\n"
        f"{context.content}"
    )


def build_rag_context(
    retriever: CulturalKnowledgeRetriever,
    query: str,
    *,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
    max_chars: int = 3500,
    include_metadata: bool = True,
) -> RagContextResult:
    """
    Retrieve and format context for inference prompt.
    """
    contexts = retriever.search(
        query,
        top_k=top_k,
        filters=filters,
    )

    parts = [
        format_retrieved_context(
            context,
            index=index,
            include_metadata=include_metadata,
        )
        for index, context in enumerate(contexts, start=1)
    ]

    context_block = "\n\n".join(parts).strip()

    if max_chars and len(context_block) > max_chars:
        context_block = context_block[:max_chars].rstrip()

    return RagContextResult(
        query=query,
        contexts=contexts,
        context_block=context_block,
    )
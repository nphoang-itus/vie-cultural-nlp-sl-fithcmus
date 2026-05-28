"""
Build prompt-ready RAG context blocks from retrieved documents.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from src.rag.retriever import CulturalKnowledgeRetriever, RetrievedContext


_VIETNAMESE_STOPWORDS = {
    "ban",
    "biet",
    "chi",
    "cho",
    "co",
    "cua",
    "duoc",
    "gi",
    "gioi",
    "hay",
    "hieu",
    "la",
    "muon",
    "toi",
    "tim",
    "ve",
    "viet",
    "nam",
}

_TET_FOOD_TERMS = {
    "am",
    "banh",
    "do",
    "mon",
    "mut",
    "thuc",
}


@dataclass(frozen=True)
class RagContextResult:
    query: str
    contexts: list[RetrievedContext]
    context_block: str


def _normalize_for_matching(text: str) -> str:
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D").lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return " ".join(text.split())


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _normalize_for_matching(text).split()
        if token and token not in _VIETNAMESE_STOPWORDS
    }


def _is_general_tet_query(query_tokens: set[str]) -> bool:
    return "tet" in query_tokens and not query_tokens.intersection(_TET_FOOD_TERMS)


def _context_sort_key(query: str, context: RetrievedContext) -> tuple[float, float]:
    """
    Keep vector score primary, but avoid sending narrow Tết contexts first when
    the user asks a broad Tết question.
    """
    query_tokens = _tokens(query)
    keyword = str(context.metadata.get("keyword") or "")
    category = str(context.metadata.get("category") or "")
    keyword_norm = _normalize_for_matching(keyword)

    score = float(context.score or 0.0)
    bonus = 0.0

    if _is_general_tet_query(query_tokens):
        if keyword_norm == "tet nguyen dan":
            bonus += 0.25
        elif category == "le_hoi":
            bonus += 0.12

        if keyword_norm in {"mut tet", "tet han thuc", "le hoi banh chung"}:
            bonus -= 0.12

    return score + bonus, score


def _deduplicate_by_keyword(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
    seen: set[str] = set()
    deduped: list[RetrievedContext] = []

    for context in contexts:
        keyword = _normalize_for_matching(str(context.metadata.get("keyword") or ""))
        key = keyword or context.doc_id
        if key in seen:
            continue
        seen.add(key)
        deduped.append(context)

    return deduped


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
    contexts = sorted(
        _deduplicate_by_keyword(contexts),
        key=lambda context: _context_sort_key(query, context),
        reverse=True,
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

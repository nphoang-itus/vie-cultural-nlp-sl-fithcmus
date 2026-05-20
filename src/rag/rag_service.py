"""
High-level RAG service for Vietnamese Cultural text-only QA.

This service is the integration point between:
- user question / normalized question
- retriever
- context builder
- text-only prompt builder
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.rag.context_builder import RagContextResult, build_rag_context
from src.rag.prompt_builder import build_text_qa_rag_prompt
from src.rag.retriever import CulturalKnowledgeRetriever


@dataclass(frozen=True)
class RagQAInput:
    """
    Input schema for text-only Vietnamese Cultural QA.

    Attributes:
        question:
            Original user question.

        normalized_question:
            Optional cleaned/rewritten question used for retrieval.
            If missing, the original question is used.

        category:
            Optional cultural category used for metadata filtering.
            Example: "am_thuc", "kien_truc".

        keyword:
            Optional cultural keyword used for metadata filtering.
            Example: "bánh chưng", "áo dài".
    """

    question: str
    normalized_question: str | None = None
    category: str | None = None
    keyword: str | None = None


@dataclass(frozen=True)
class RagQAOutput:
    """
    Output schema after preparing RAG context and prompt.
    """

    retrieval_query: str
    filters: dict[str, Any] | None
    rag_context: RagContextResult
    prompt: str


# Temporary backward-compatible alias.
# Remove after all old imports are migrated.
RagInferenceInput = RagQAInput
RagInferenceOutput = RagQAOutput


class RagService:
    """
    Main RAG service used by the text-only QA pipeline.
    """

    def __init__(
        self,
        retriever: CulturalKnowledgeRetriever,
        *,
        top_k: int = 5,
        max_context_chars: int = 3500,
        use_category_filter: bool = False,
        use_keyword_filter: bool = False,
    ):
        self.retriever = retriever
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.use_category_filter = use_category_filter
        self.use_keyword_filter = use_keyword_filter

    @classmethod
    def from_config_file(
        cls,
        config_path: Path = Path("configs/rag.yaml"),
    ) -> "RagService":
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        retriever = CulturalKnowledgeRetriever.from_config_dict(config)
        inference_cfg = config.get("inference", {})

        return cls(
            retriever=retriever,
            top_k=int(inference_cfg.get("top_k", 5)),
            max_context_chars=int(inference_cfg.get("max_context_chars", 3500)),
            use_category_filter=bool(inference_cfg.get("use_category_filter", False)),
            use_keyword_filter=bool(inference_cfg.get("use_keyword_filter", False)),
        )

    def prepare_prompt(self, data: RagQAInput) -> RagQAOutput:
        """
        Prepare a text-only RAG prompt with retrieved cultural context.

        Prefer normalized_question for retrieval if provided.
        Fallback to original question otherwise.
        """
        retrieval_query = (
            str(data.normalized_question or "").strip()
            or str(data.question or "").strip()
        )

        if not retrieval_query:
            raise ValueError("Both question and normalized_question are empty.")

        filters = self._build_filters(data)

        rag_context = build_rag_context(
            retriever=self.retriever,
            query=retrieval_query,
            top_k=self.top_k,
            filters=filters,
            max_chars=self.max_context_chars,
            include_metadata=True,
        )

        prompt = build_text_qa_rag_prompt(
            question=retrieval_query,
            rag_context=rag_context.context_block,
        )

        return RagQAOutput(
            retrieval_query=retrieval_query,
            filters=filters,
            rag_context=rag_context,
            prompt=prompt,
        )

    def _build_filters(self, data: RagQAInput) -> dict[str, Any] | None:
        conditions: list[dict[str, Any]] = []

        if self.use_category_filter and data.category:
            conditions.append({"category": data.category})

        if self.use_keyword_filter and data.keyword:
            conditions.append({"keyword": data.keyword})

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]

        return {"$and": conditions}
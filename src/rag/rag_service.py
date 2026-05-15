"""
High-level RAG service for inference pipeline.

This service is the integration point between:
- standalone question
- retriever
- context builder
- prompt builder
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.rag.context_builder import RagContextResult, build_rag_context
from src.rag.prompt_builder import build_vqa_rag_prompt
from src.rag.retriever import CulturalKnowledgeRetriever

import yaml


@dataclass(frozen=True)
class RagInferenceInput:
    question: str
    standalone_question: str | None = None
    vision_caption: str | None = None
    category: str | None = None
    keyword: str | None = None


@dataclass(frozen=True)
class RagInferenceOutput:
    retrieval_query: str
    filters: dict[str, Any] | None
    rag_context: RagContextResult
    prompt: str


class RagService:
    """
    Main service used by inference pipeline.
    """

    def __init__(
        self,
        retriever: CulturalKnowledgeRetriever,
        *,
        top_k: int = 5,
        max_context_chars: int = 3500,
        use_category_filter: bool = True,
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
            use_category_filter=bool(inference_cfg.get("use_category_filter", True)),
            use_keyword_filter=bool(inference_cfg.get("use_keyword_filter", False)),
        )

    def prepare_prompt(self, data: RagInferenceInput) -> RagInferenceOutput:
        """
        Prepare prompt with retrieved context.

        Prefer standalone_question for retrieval because it is less ambiguous.
        Fallback to original question if standalone_question is missing.
        """
        retrieval_query = (
            str(data.standalone_question or "").strip()
            or str(data.question or "").strip()
        )

        if not retrieval_query:
            raise ValueError("Both question and standalone_question are empty.")

        filters = self._build_filters(data)

        rag_context = build_rag_context(
            retriever=self.retriever,
            query=retrieval_query,
            top_k=self.top_k,
            filters=filters,
            max_chars=self.max_context_chars,
            include_metadata=True,
        )

        prompt = build_vqa_rag_prompt(
            question=retrieval_query,
            rag_context=rag_context.context_block,
            vision_caption=data.vision_caption,
        )

        return RagInferenceOutput(
            retrieval_query=retrieval_query,
            filters=filters,
            rag_context=rag_context,
            prompt=prompt,
        )

    def _build_filters(self, data: RagInferenceInput) -> dict[str, Any] | None:
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
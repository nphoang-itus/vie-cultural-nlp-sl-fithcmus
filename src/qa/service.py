"""
End-to-end Vietnamese Cultural QA service.

This service orchestrates:
- RAG prompt preparation
- Qwen LoRA answer generation
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from src.llm.qwen_lora_generator import QwenLoraGenerator
from src.qa.schemas import QARequest, QAResponse, RetrievedContextInfo
from src.rag.rag_service import RagQAInput, RagService


@dataclass(frozen=True)
class QATiming:
    """
    Runtime timing for debugging.
    """

    rag_seconds: float
    generation_seconds: float
    total_seconds: float


class QAService:
    """
    End-to-end QA pipeline.

    Main method:
        answer(request: QARequest) -> QAResponse
    """

    def __init__(
        self,
        rag_service: RagService,
        generator: QwenLoraGenerator,
    ):
        self.rag_service = rag_service
        self.generator = generator

    @classmethod
    def from_config_file(
        cls,
        config_path: Path = Path("configs/rag.yaml"),
    ) -> "QAService":
        """
        Build QAService from config.

        This loads:
        - Retriever / vector store / embedding through RagService
        - Qwen base model + LoRA adapter through QwenLoraGenerator
        """
        rag_service = RagService.from_config_file(config_path)
        generator = QwenLoraGenerator.from_config_file(config_path)

        return cls(
            rag_service=rag_service,
            generator=generator,
        )

    def answer(self, request: QARequest) -> QAResponse:
        """
        Generate an answer for one Vietnamese cultural question.
        """
        question = str(request.question or "").strip()

        if not question:
            raise ValueError("question cannot be empty.")

        rag_input = RagQAInput(
            question=question,
            normalized_question=request.normalized_question,
            category=request.category,
            keyword=request.keyword,
        )

        rag_output = self.rag_service.prepare_prompt(rag_input)

        answer = self.generator.generate(rag_output.prompt)

        contexts = [
            RetrievedContextInfo(
                doc_id=context.doc_id,
                content=context.content,
                metadata=context.metadata,
                distance=context.distance,
                score=context.score,
            )
            for context in rag_output.rag_context.contexts
        ]

        return QAResponse(
            question=question,
            retrieval_query=rag_output.retrieval_query,
            answer=answer,
            prompt=rag_output.prompt,
            filters=rag_output.filters,
            contexts=contexts,
        )

    def answer_with_timing(self, request: QARequest) -> tuple[QAResponse, QATiming]:
        """
        Same as answer(), but also returns runtime timing.

        Useful for local performance testing on Mac MPS.
        """
        total_start = time.time()

        question = str(request.question or "").strip()

        if not question:
            raise ValueError("question cannot be empty.")

        rag_input = RagQAInput(
            question=question,
            normalized_question=request.normalized_question,
            category=request.category,
            keyword=request.keyword,
        )

        rag_start = time.time()
        rag_output = self.rag_service.prepare_prompt(rag_input)
        rag_seconds = time.time() - rag_start

        gen_start = time.time()
        answer = self.generator.generate(rag_output.prompt)
        generation_seconds = time.time() - gen_start

        contexts = [
            RetrievedContextInfo(
                doc_id=context.doc_id,
                content=context.content,
                metadata=context.metadata,
                distance=context.distance,
                score=context.score,
            )
            for context in rag_output.rag_context.contexts
        ]

        response = QAResponse(
            question=question,
            retrieval_query=rag_output.retrieval_query,
            answer=answer,
            prompt=rag_output.prompt,
            filters=rag_output.filters,
            contexts=contexts,
        )

        timing = QATiming(
            rag_seconds=rag_seconds,
            generation_seconds=generation_seconds,
            total_seconds=time.time() - total_start,
        )

        return response, timing
"""
Schemas for Vietnamese Cultural text-only QA pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QARequest:
    """
    User-facing input for text-only Vietnamese Cultural QA.

    Attributes:
        question:
            Original Vietnamese user question.

        normalized_question:
            Optional rewritten/normalized question for retrieval.
            If not provided, question is used directly.

        category:
            Optional cultural category for retrieval filtering.
            Example: "am_thuc", "kien_truc", "le_hoi".s

        keyword:
            Optional cultural keyword for retrieval filtering.
            Example: "bánh chưng", "áo dài".
    """

    question: str
    normalized_question: str | None = None
    category: str | None = None
    keyword: str | None = None


@dataclass(frozen=True)
class RetrievedContextInfo:
    """
    Lightweight context information returned for debugging/evaluation.
    """

    doc_id: str
    content: str
    metadata: dict[str, Any]
    distance: float | None
    score: float | None


@dataclass(frozen=True)
class QAResponse:
    """
    Final output from the QA pipeline.
    """

    question: str
    retrieval_query: str
    answer: str
    prompt: str
    filters: dict[str, Any] | None
    contexts: list[RetrievedContextInfo]
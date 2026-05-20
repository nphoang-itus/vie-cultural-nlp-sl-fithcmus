from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QAApiRequest(BaseModel):
    question: str = Field(..., min_length=1)
    normalized_question: str | None = None
    category: str | None = None
    keyword: str | None = None
    debug: bool = True


class RetrievedContextApiResponse(BaseModel):
    doc_id: str
    content: str
    metadata: dict[str, Any]
    distance: float | None
    score: float | None


class QATimingApiResponse(BaseModel):
    rag_seconds: float
    generation_seconds: float
    total_seconds: float


class QAApiResponse(BaseModel):
    question: str
    retrieval_query: str
    answer: str
    timing: QATimingApiResponse

    filters: dict[str, Any] | None = None
    contexts: list[RetrievedContextApiResponse] = []

    # Only return when debug=True
    prompt: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
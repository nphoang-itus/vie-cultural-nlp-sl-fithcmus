from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    HealthResponse,
    QAApiRequest,
    QAApiResponse,
    QATimingApiResponse,
    RetrievedContextApiResponse,
)
from src.qa.schemas import QARequest
from src.qa.service import QAService

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/rag.yaml")

# Local LLM generation should usually be serialized.
# This avoids multiple requests fighting for MPS/GPU memory.
generation_lock = Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading QAService from config: %s", CONFIG_PATH)

    try:
        app.state.qa_service = QAService.from_config_file(CONFIG_PATH)
        app.state.model_loaded = True
        logger.info("QAService loaded successfully.")
    except Exception:
        app.state.qa_service = None
        app.state.model_loaded = False
        logger.exception("Failed to load QAService.")
        raise

    yield

    logger.info("Shutting down QA API.")


app = FastAPI(
    title="Vietnamese Cultural QA API",
    description="Text-only RAG + Qwen2.5 LoRA QA API",
    version="0.1.0",
    lifespan=lifespan,
)

# Useful later when Streamlit/React calls this API from another port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok" if request.app.state.model_loaded else "not_ready",
        model_loaded=bool(request.app.state.model_loaded),
    )


@app.post("/api/qa", response_model=QAApiResponse)
def answer_question(
    payload: QAApiRequest,
    request: Request,
) -> QAApiResponse:
    qa_service: QAService | None = request.app.state.qa_service

    if qa_service is None:
        raise HTTPException(
            status_code=503,
            detail="QA service is not loaded.",
        )

    question = payload.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="question cannot be empty.",
        )

    qa_request = QARequest(
        question=question,
        normalized_question=_clean_optional(payload.normalized_question),
        category=_clean_optional(payload.category),
        keyword=_clean_optional(payload.keyword),
    )

    try:
        with generation_lock:
            qa_response, timing = qa_service.answer_with_timing(qa_request)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Failed to answer question.")
        raise HTTPException(
            status_code=500,
            detail=f"Internal QA error: {exc}",
        ) from exc

    contexts = [
        RetrievedContextApiResponse(
            doc_id=context.doc_id,
            content=context.content,
            metadata=context.metadata,
            distance=context.distance,
            score=context.score,
        )
        for context in qa_response.contexts
    ]

    return QAApiResponse(
        question=qa_response.question,
        retrieval_query=qa_response.retrieval_query,
        answer=qa_response.answer,
        timing=QATimingApiResponse(
            rag_seconds=timing.rag_seconds,
            generation_seconds=timing.generation_seconds,
            total_seconds=timing.total_seconds,
        ),
        filters=qa_response.filters,
        contexts=contexts if payload.debug else [],
        prompt=qa_response.prompt if payload.debug else None,
    )


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None

    value = value.strip()
    return value or None
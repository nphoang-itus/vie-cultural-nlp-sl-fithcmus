"""
Standalone question service - orchestration layer.
Combines template_rewriter and llm_rewriter.
This is the single entry point for the standalone question step.
"""

import logging
from typing import Any

from src.standalone.template_rewriter import rewrite as template_rewrite
from src.standalone.llm_rewriter import rewrite_with_llm

logger = logging.getLogger(__name__)


def process_record(
    record: dict[str, Any],
    use_llm_fallback: bool = False,
    llm_model: str = "gemini-2.0-flash",
    llm_max_retries: int = 2,
    llm_timeout: int = 10,
) -> dict[str, Any]:
    """
    Add standalone_question and rewrite_method to one record.
    Returns a new dict - never mutates the input.
    """
    keyword: str = record.get("keyword", "").strip()
    vision_caption: str = record.get("vision_caption", "").strip()
    question: str = record.get("question", "").strip()

    # Layer 1: template rewrite by keyword.
    template_result = template_rewrite(keyword, question)

    if template_result.matched:
        return {
            **record,
            "standalone_question": template_result.standalone_question,
            "rewrite_method": "template",
        }

    # Layer 2: LLM fallback using vision_caption + question.
    if use_llm_fallback:
        llm_result = rewrite_with_llm(
            vision_caption=vision_caption,
            question=question,
            keyword=keyword,
            model=llm_model,
            max_retries=llm_max_retries,
            timeout=llm_timeout,
        )

        if llm_result.success:
            return {
                **record,
                "standalone_question": llm_result.standalone_question,
                "rewrite_method": "llm",
            }

        logger.warning(f"LLM failed for question: {question!r}")
        return {
            **record,
            "standalone_question": question,
            "rewrite_method": "failed",
        }

    # Layer 3: keep original.
    return {
        **record,
        "standalone_question": question,
        "rewrite_method": "unchanged",
    }


def process_records(
    records: list[dict[str, Any]],
    use_llm_fallback: bool = False,
    **llm_kwargs,
) -> list[dict[str, Any]]:
    """Process a list of records, logging progress every 500 rows."""
    results = []
    for i, record in enumerate(records):
        if i % 500 == 0:
            logger.info(f"Processing record {i}/{len(records)}...")
        results.append(process_record(record, use_llm_fallback, **llm_kwargs))
    return results
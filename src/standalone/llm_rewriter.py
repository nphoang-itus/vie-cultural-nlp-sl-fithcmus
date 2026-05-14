"""
LLM-based standalone question rewriter (optional fallback).
Uses Gemini API. Disabled cleanly when use_llm_fallback=false in config
or when GEMINI_API_KEY is not set. Swap the provider here without
touching any other module.
"""

import logging
import time
from typing import NamedTuple

logger = logging.getLogger(__name__)


class LLMRewriteResult(NamedTuple):
    standalone_question: str
    success: bool

# [TODO] - Viết lại câu prompt để bao phủ toàn bộ case
_PROMPT_TEMPLATE = """You are rewriting Visual Question Answering questions into standalone Vietnamese questions.

Vision caption:
{vision_caption}

Original question:
{question}

Rules:
- Rewrite the original question into a complete Vietnamese standalone question.
- Use the visual subject from the vision caption as the explicit subject of the question.
- If the vision caption is in English, translate only the necessary visual subject into Vietnamese before rewriting.
- If the vision caption is already Vietnamese, keep the rewritten question in Vietnamese.
- Preserve the original meaning of the question.
- Do not answer the question.
- Do not add cultural facts that are not present in the question or caption.
- Return only the rewritten question.

Standalone question:"""

# [TODO] - Tìm phương án tối ưu số lần call API bằng cách gộp nhóm các câu hỏi và truyền bất đồng bộ để gom thành 1 câu prompt
def rewrite_with_llm(
    vision_caption: str,
    question: str,
    model: str = "gemini-2.5-flash-lite",
    max_retries: int = 2,
    timeout: int = 10,
) -> LLMRewriteResult:
    """
    Call Gemini API to rewrite one question.
    Returns success=False on any error.
    """
    try:
        from google import genai
        from google.genai import types
        from src.utils.config import get_env

        api_key = get_env("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set - skipping LLM rewrite.")
            return LLMRewriteResult(standalone_question=question, success=False)

        client = genai.Client(api_key=api_key)

        prompt = _PROMPT_TEMPLATE.format(
            vision_caption=vision_caption,
            question=question,
        )

        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=128,
                    ),
                )

                rewritten = (response.text or "").strip()

                if rewritten:
                    return LLMRewriteResult(
                        standalone_question=rewritten,
                        success=True,
                    )

            except Exception as e:
                logger.warning(f"LLM attempt {attempt}/{max_retries} failed: {e}")
                time.sleep(1)

        return LLMRewriteResult(standalone_question=question, success=False)

    except ImportError:
        logger.warning("google-genai not installed — LLM rewrite unavailable.")
        return LLMRewriteResult(standalone_question=question, success=False)
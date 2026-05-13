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
_PROMPT_TEMPLATE = """You are rewriting Vietnamese VQA questions to be standalone.

The cultural subject is: {keyword}

Original question: {question}

Rules:
- Replace demonstrative pronouns (này, đây, đó) with the keyword.
- Keep the question meaning identical — do not add extra information.
- Return ONLY the rewritten question. No explanation, no punctuation changes.

Rewritten question:"""

# [TODO] - Tìm phương án tối ưu số lần call API bằng cách gộp nhóm các câu hỏi và truyền bất đồng bộ để gom thành 1 câu prompt
def rewrite_with_llm(
    keyword: str,
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

        prompt = _PROMPT_TEMPLATE.format(keyword=keyword, question=question)

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
"""
LLM-based standalone question rewriter (optional fallback).
Uses Gemini API. Disabled cleanly when use_llm_fallback=false in config
or when GEMINI_API_KEY is not set. Swap the provider here without
touching any other module.
"""

import logging
import time
from typing import NamedTuple
import json
import re

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
    
def _strip_markdown_code_fence(text: str) -> str:
    """Remove ```json ... ``` wrapper if Gemini returns markdown."""
    text = (text or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    return text


def _parse_json_array(text: str) -> list[dict]:
    """Parse a JSON array from model output."""
    cleaned = _strip_markdown_code_fence(text)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Cannot parse LLM batch output as JSON: {e}\nOutput: {text[:500]}")

    if not isinstance(parsed, list):
        raise ValueError("LLM batch output is not a JSON array.")

    return parsed

_BATCH_PROMPT_TEMPLATE = """You are rewriting Visual Question Answering questions into standalone Vietnamese questions.

You will receive a JSON array of items. Each item contains:
- index: integer
- vision_caption: image caption, may be English or Vietnamese
- question: original Vietnamese question

Task:
Rewrite each question into a complete Vietnamese standalone question.

Rules:
- Use the visual subject from the vision_caption as the explicit subject.
- If the vision_caption is English, translate only the necessary visual subject into Vietnamese.
- Preserve the original meaning of the question.
- Do not answer the question.
- Do not add cultural facts that are not present in the question or caption.
- Return ONLY a valid JSON array.
- Do not wrap the JSON in markdown.
- Keep the same index values.

Output format:
[
    {{"index": 0, "standalone_question": "..."}},
    {{"index": 1, "standalone_question": "..."}}
]

Input items:
{items_json}

Output:
"""

class LLMBatchRewriteResult(NamedTuple):
    results: dict[int, str]
    success: bool

def rewrite_batch_with_llm(
    items: list[dict],
    model: str = "gemini-2.5-flash-lite",
    max_retries: int = 2,
    timeout: int = 10,
) -> LLMBatchRewriteResult:
    """
    Call Gemini API to rewrite a batch of questions.

    Input item format:
        {
            "index": int,
            "vision_caption": str,
            "question": str
        }

    Returns:
        results: {index: standalone_question}
    """
    try:
        from google import genai
        from google.genai import types
        from src.utils.config import get_env

        api_key = get_env("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set - skipping batch LLM rewrite.")
            return LLMBatchRewriteResult(results={}, success=False)

        if not items:
            return LLMBatchRewriteResult(results={}, success=True)

        client = genai.Client(api_key=api_key)

        compact_items = [
            {
                "index": int(item["index"]),
                "vision_caption": str(item.get("vision_caption", "")).strip(),
                "question": str(item.get("question", "")).strip(),
            }
            for item in items
        ]

        items_json = json.dumps(compact_items, ensure_ascii=False)

        prompt = _BATCH_PROMPT_TEMPLATE.format(items_json=items_json)

        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=4096,
                    ),
                )

                raw_text = (response.text or "").strip()

                parsed_items = _parse_json_array(raw_text)

                results: dict[int, str] = {}

                for parsed in parsed_items:
                    index = parsed.get("index")
                    standalone_question = str(parsed.get("standalone_question", "")).strip()

                    if isinstance(index, int) and standalone_question:
                        results[index] = standalone_question

                if results:
                    return LLMBatchRewriteResult(results=results, success=True)

            except Exception as e:
                logger.warning(f"Batch LLM attempt {attempt}/{max_retries} failed: {e}")
                time.sleep(1)

        return LLMBatchRewriteResult(results={}, success=False)

    except ImportError:
        logger.warning("google-genai not installed — batch LLM rewrite unavailable.")
        return LLMBatchRewriteResult(results={}, success=False)
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

_PROMPT_TEMPLATE = """You are rewriting a Visual Question Answering question into a concise standalone Vietnamese question.

Keyword:
{keyword}

Vision caption:
{vision_caption}

Original question:
{question}

Task:
Rewrite the original question into a concise, natural Vietnamese standalone question.

Rules:
- Prefer the keyword as the main subject.
- Use vision_caption only to resolve vague references such as "đây", "này", "nó", "hình ảnh này", "món này".
- If the original question already contains a clear subject, keep it almost unchanged.
- Do not copy long visual descriptions from the caption.
- Do not include colors, layout, background, camera angle, or surrounding objects unless they are essential to the question.
- Do not answer the question.
- Do not add cultural facts.
- Keep the question concise, ideally under 25 Vietnamese words.
- Return only the rewritten question.

Standalone question:"""

def rewrite_with_llm(
    vision_caption: str,
    question: str,
    keyword: str = "",
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
            keyword=keyword.strip(),
            vision_caption=vision_caption.strip(),
            question=question.strip(),
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

_BATCH_PROMPT_TEMPLATE = """You are rewriting Visual Question Answering questions into concise standalone Vietnamese questions.

You will receive a JSON array of items. Each item contains:
- index: integer
- keyword: short cultural subject, may be Vietnamese without accents or mixed English/Vietnamese
- vision_caption: image caption, may be English or Vietnamese
- question: original Vietnamese question

Task:
Rewrite each original question into a concise, natural Vietnamese standalone question.

Core principle:
A standalone question should be understandable without seeing the image, but it must NOT copy long visual descriptions from the caption.

Rules:
1. Prefer the keyword as the main subject of the rewritten question.
2. Use vision_caption only to resolve vague references such as "đây", "này", "nó", "hình ảnh này", "món này", "trang phục này", "công trình này".
3. If the original question already contains a clear subject, keep it almost unchanged.
4. Do NOT copy full caption details such as colors, layout, background, camera angle, surrounding objects, or long object lists.
5. Do NOT add cultural facts or answer the question.
6. Keep the rewritten question concise, ideally under 25 Vietnamese words.
7. The output must be natural Vietnamese.
8. Return ONLY a valid JSON array. Do not wrap the JSON in markdown.
9. Keep the same index values.

Keyword normalization guidance:
- If keyword is Vietnamese without accents, convert it to natural Vietnamese when obvious.
- Examples:
  - "banh chung Tet" -> "bánh chưng Tết"
  - "ao dai" -> "áo dài"
  - "non la" -> "nón lá"
- If unsure, keep the keyword but make the sentence grammatical.

Examples:

Input:
[
  {{
    "index": 0,
    "keyword": "banh chung Tet",
    "vision_caption": "Hình ảnh chụp cận cảnh hai chiếc bánh chưng được gói trong lá dong, đặt trên đĩa trắng, xung quanh có hoa mai và pháo giấy.",
    "question": "Ý nghĩa văn hóa của bánh chưng là gì?"
  }},
  {{
    "index": 1,
    "keyword": "banh chung Tet",
    "vision_caption": "A square sticky rice cake wrapped in green leaves, commonly seen during Vietnamese Lunar New Year.",
    "question": "Nó có ý nghĩa gì trong ngày Tết?"
  }},
  {{
    "index": 2,
    "keyword": "ao dai",
    "vision_caption": "A Vietnamese woman wearing a traditional long dress.",
    "question": "Trang phục này tên gì?"
  }}
]

Output:
[
  {{"index": 0, "standalone_question": "Ý nghĩa văn hóa của bánh chưng là gì?"}},
  {{"index": 1, "standalone_question": "Bánh chưng có ý nghĩa gì trong ngày Tết?"}},
  {{"index": 2, "standalone_question": "Áo dài là trang phục gì?"}}
]

Now rewrite the following input items.

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
                "keyword": str(item.get("keyword", "")).strip(),
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
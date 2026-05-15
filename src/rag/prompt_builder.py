"""
Prompt builder for RAG-enhanced Vietnamese Cultural VQA.
"""

from __future__ import annotations


def build_vqa_rag_prompt(
    *,
    question: str,
    rag_context: str,
    vision_caption: str | None = None,
) -> str:
    """
    Build a Vietnamese prompt for answer generation.

    Args:
        question: standalone question or user question.
        rag_context: retrieved cultural context block.
        vision_caption: optional image description/caption.

    Returns:
        Prompt string.
    """
    question = str(question or "").strip()
    rag_context = str(rag_context or "").strip()
    vision_caption = str(vision_caption or "").strip()

    image_part = ""
    if vision_caption:
        image_part = f"\nMô tả hình ảnh:\n{vision_caption}\n"

    # [TODO]: Tối ưu lại câu prompt để truy xuất tốt hơn
    return f"""Bạn là hệ thống trả lời câu hỏi về văn hóa Việt Nam dựa trên hình ảnh và tri thức truy xuất được.

Nhiệm vụ:
- Trả lời bằng tiếng Việt.
- Ưu tiên sử dụng thông tin trong phần Ngữ cảnh văn hóa.
- Nếu ngữ cảnh không đủ, hãy trả lời thận trọng, không bịa chi tiết.
- Câu trả lời nên ngắn gọn, đúng trọng tâm, phù hợp với câu hỏi.

Câu hỏi:
{question}
{image_part}
Ngữ cảnh văn hóa:
{rag_context}

Câu trả lời:""".strip()
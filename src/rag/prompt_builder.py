"""
Prompt builder for RAG-enhanced Vietnamese Cultural Question Answering.
"""

from __future__ import annotations


def build_text_qa_rag_prompt(
    *,
    question: str,
    rag_context: str,
) -> str:
    """
    Build a Vietnamese text-only RAG prompt for cultural question answering.

    Args:
        question: User question or normalized retrieval question.
        rag_context: Retrieved cultural context block.

    Returns:
        Prompt string for answer generation.
    """
    question = str(question or "").strip()
    rag_context = str(rag_context or "").strip()

    if not question:
        raise ValueError("question cannot be empty.")

    if not rag_context:
        rag_context = (
            "Không có ngữ cảnh truy xuất phù hợp. "
            "Hãy trả lời thận trọng và nói rõ nếu không đủ thông tin."
        )

    return f"""Bạn là trợ lý trả lời câu hỏi về văn hóa Việt Nam dựa trên tri thức truy xuất được.

Nhiệm vụ:
- Trả lời bằng tiếng Việt.
- Ưu tiên sử dụng thông tin trong phần Ngữ cảnh văn hóa.
- Nếu ngữ cảnh không đủ, hãy nói rõ rằng thông tin truy xuất chưa đủ.
- Không bịa thêm chi tiết ngoài ngữ cảnh nếu không chắc chắn.
- Câu trả lời nên ngắn gọn, đúng trọng tâm và phù hợp với câu hỏi.

Câu hỏi:
{question}

Ngữ cảnh văn hóa:
{rag_context}

Câu trả lời:""".strip()


# Backward-compatible alias.
# Keep this temporarily if older code still imports build_vqa_rag_prompt.
def build_vqa_rag_prompt(
    *,
    question: str,
    rag_context: str,
    vision_caption: str | None = None,
) -> str:
    """
    Deprecated compatibility wrapper.

    The project has been migrated from Visual QA to text-only QA.
    vision_caption is intentionally ignored.
    """
    return build_text_qa_rag_prompt(
        question=question,
        rag_context=rag_context,
    )
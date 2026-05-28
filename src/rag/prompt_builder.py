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
        return f"""Bạn là trợ lý trả lời câu hỏi về văn hóa Việt Nam.

Không có ngữ cảnh văn hóa đủ liên quan được truy xuất từ cơ sở tri thức.

Nhiệm vụ:
- Trả lời bằng tiếng Việt.
- Không suy đoán hoặc bịa chi tiết văn hóa khi không có ngữ cảnh đáng tin cậy.
- Đây là bài toán hỏi đáp văn bản, không có ảnh đầu vào. Không dùng các cụm như "ảnh", "hình ảnh", "trong ảnh", "ảnh chụp", "nhìn thấy", "cho thấy" hoặc mô tả thị giác.
- Nếu câu hỏi nằm ngoài phạm vi văn hóa Việt Nam hoặc quá chung chung, hãy nói ngắn gọn rằng bạn có thể hỗ trợ trả lời các câu hỏi về văn hóa Việt Nam.
- Nếu câu hỏi có vẻ thuộc phạm vi văn hóa Việt Nam nhưng thiếu dữ liệu truy xuất, hãy nói rằng bạn chưa có đủ thông tin trong cơ sở tri thức để trả lời chắc chắn.

Câu hỏi:
{question}""".strip()

    return f"""Bạn là trợ lý trả lời câu hỏi về văn hóa Việt Nam dựa trên tri thức truy xuất được.

Nhiệm vụ:
- Trả lời bằng tiếng Việt.
- Chỉ sử dụng các thông tin có trong phần Ngữ cảnh văn hóa khi trả lời nội dung văn hóa cụ thể.
- Đây là bài toán hỏi đáp văn bản, không có ảnh đầu vào. Không dùng các cụm như "ảnh", "hình ảnh", "trong ảnh", "ảnh chụp", "nhìn thấy", "cho thấy" hoặc mô tả thị giác.
- Không mặc định context đầu tiên là đúng nhất; hãy tổng hợp các context liên quan trực tiếp nhất với câu hỏi và ưu tiên chủ đề tổng quát khi câu hỏi mang tính tìm hiểu chung.
- Nếu ngữ cảnh không trực tiếp hỗ trợ câu trả lời, hãy nói rõ rằng thông tin truy xuất chưa đủ.
- Không bịa thêm chi tiết ngoài ngữ cảnh nếu không chắc chắn.
- Câu trả lời nên ngắn gọn, đúng trọng tâm và phù hợp với câu hỏi.

Câu hỏi:
{question}

Ngữ cảnh văn hóa:
{rag_context}""".strip()


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

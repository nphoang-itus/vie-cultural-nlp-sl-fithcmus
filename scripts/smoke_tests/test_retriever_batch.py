"""
Batch smoke test for retriever.

Usage:
  python scripts/test_retriever_batch.py --config configs/rag.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rag.retriever import CulturalKnowledgeRetriever


QUERIES = [
    "Bánh chưng có ý nghĩa gì trong ngày Tết?",
    "Áo dài thể hiện nét đẹp văn hóa nào?",
    "Chùa Một Cột có đặc điểm kiến trúc gì?",
    "Đàn bầu có vai trò gì trong âm nhạc truyền thống Việt Nam?",
    "Lễ hội Đền Hùng gắn với truyền thống nào?",
    "Môn đẩy gậy có nguồn gốc và ý nghĩa gì?",
]


def main() -> None:
    retriever = CulturalKnowledgeRetriever.from_config_file(
        Path("configs/rag.yaml")
    )

    for query in QUERIES:
        print("=" * 100)
        print("Query:", query)

        results = retriever.search(query, top_k=3)

        for rank, item in enumerate(results, start=1):
            print(
                f"{rank}. keyword={item.metadata.get('keyword')} | "
                f"category={item.metadata.get('category')} | "
                f"score={item.score} | "
                f"doc_id={item.doc_id}"
            )


if __name__ == "__main__":
    main()
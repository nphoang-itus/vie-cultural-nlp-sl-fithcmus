"""
Test RAG integration for inference prompt preparation.

Usage:
  python scripts/test_rag_inference.py \
    --config configs/rag.yaml \
    --question "Nó có ý nghĩa gì trong ngày Tết?" \
    --standalone-question "Bánh chưng có ý nghĩa gì trong ngày Tết?" \
    --category am_thuc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.rag_service import RagInferenceInput, RagService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test RAG inference integration.")

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rag.yaml"),
    )

    parser.add_argument(
        "--question",
        type=str,
        default="Nó có ý nghĩa gì trong ngày Tết?",
    )

    parser.add_argument(
        "--standalone-question",
        type=str,
        default="Bánh chưng có ý nghĩa gì trong ngày Tết?",
    )

    parser.add_argument(
        "--vision-caption",
        type=str,
        default="Hình ảnh cho thấy một chiếc bánh chưng được gói bằng lá dong.",
    )

    parser.add_argument(
        "--category",
        type=str,
        default="am_thuc",
    )

    parser.add_argument(
        "--keyword",
        type=str,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    service = RagService.from_config_file(args.config)

    output = service.prepare_prompt(
        RagInferenceInput(
            question=args.question,
            standalone_question=args.standalone_question,
            vision_caption=args.vision_caption,
            category=args.category,
            keyword=args.keyword,
        )
    )

    print("retrieval_query:", output.retrieval_query)
    print("filters:", output.filters)
    print("retrieved_contexts:", len(output.rag_context.contexts))

    print("\nTop retrieved contexts:")
    for idx, ctx in enumerate(output.rag_context.contexts, start=1):
        print("-" * 80)
        print("rank:", idx)
        print("doc_id:", ctx.doc_id)
        print("score:", ctx.score)
        print("distance:", ctx.distance)
        print("keyword:", ctx.metadata.get("keyword"))
        print("category:", ctx.metadata.get("category"))
        print("content_head:", ctx.content[:220])

    print("\n" + "=" * 80)
    print("Prompt preview:")
    print("=" * 80)
    print(output.prompt[:3000])

    if not output.rag_context.contexts:
        raise SystemExit("No retrieved contexts.")

    print("\nRAG inference integration test passed.")


if __name__ == "__main__":
    main()
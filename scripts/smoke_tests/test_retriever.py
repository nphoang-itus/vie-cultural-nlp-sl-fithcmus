"""
Smoke test CulturalKnowledgeRetriever.

Usage:
  python scripts/test_retriever.py \
    --config configs/rag.yaml \
    --query "Bánh chưng có ý nghĩa gì trong ngày Tết?" \
    --top-k 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rag.retriever import CulturalKnowledgeRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test cultural knowledge retriever.")

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rag.yaml"),
    )

    parser.add_argument(
        "--query",
        type=str,
        default="Bánh chưng có ý nghĩa gì trong ngày Tết?",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Optional category filter, e.g. am_thuc.",
    )

    parser.add_argument(
        "--keyword",
        type=str,
        default=None,
        help="Optional exact keyword filter, e.g. bánh chưng.",
    )

    parser.add_argument(
        "--show-context-block",
        action="store_true",
    )

    return parser.parse_args()


def build_filters(args: argparse.Namespace) -> dict[str, Any] | None:
    conditions = []

    if args.category:
        conditions.append({"category": args.category})

    if args.keyword:
        conditions.append({"keyword": args.keyword})

    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    retriever = CulturalKnowledgeRetriever.from_config_dict(config)

    filters = build_filters(args)

    results = retriever.search(
        args.query,
        top_k=args.top_k,
        filters=filters,
    )

    print("Query:", args.query)
    print("Filters:", filters)
    print("Returned:", len(results))

    for rank, item in enumerate(results, start=1):
        print("-" * 80)
        print("rank:", rank)
        print("doc_id:", item.doc_id)
        print("score:", item.score)
        print("distance:", item.distance)
        print("keyword:", item.metadata.get("keyword"))
        print("category:", item.metadata.get("category"))
        print("subcategory:", item.metadata.get("subcategory"))
        print("content_head:", item.content[:300])

    if not results:
        raise SystemExit("Retriever returned no results.")

    if args.show_context_block:
        print("\n" + "=" * 80)
        print("Context block:")
        print("=" * 80)
        print(
            retriever.build_context_block(
                args.query,
                top_k=args.top_k,
                filters=filters,
                max_chars=2500,
            )
        )

    print("\nRetriever smoke test passed.")


if __name__ == "__main__":
    main()
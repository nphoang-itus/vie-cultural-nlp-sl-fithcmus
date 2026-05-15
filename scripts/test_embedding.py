"""
Smoke test embedding layer.

Usage:
  python scripts/test_embedding.py \
    --config configs/rag.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.embedding import build_embedding_model_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test embedding model.")

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rag.yaml"),
    )

    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = load_yaml(args.config)

    embedder = build_embedding_model_from_config(config)

    texts = [
        "Bánh chưng là món ăn truyền thống trong ngày Tết của người Việt.",
        "Áo dài là trang phục truyền thống gắn với văn hóa Việt Nam.",
        "Chùa Một Cột là công trình kiến trúc nổi tiếng ở Hà Nội.",
    ]

    embeddings = embedder.embed_texts(texts)
    query_embedding = embedder.embed_query("Bánh chưng có ý nghĩa gì trong ngày Tết?")

    print("model_name:", embedder.config.model_name)
    print("device:", embedder.device)
    print("embedding_dimension:", embedder.embedding_dimension)
    print("num_text_embeddings:", len(embeddings))
    print("first_embedding_length:", len(embeddings[0]))
    print("query_embedding_length:", len(query_embedding))

    if len(embeddings) != len(texts):
        raise SystemExit("Mismatch: number of embeddings != number of texts")

    if len(embeddings[0]) != embedder.embedding_dimension:
        raise SystemExit("Mismatch: embedding length != model dimension")

    if len(query_embedding) != embedder.embedding_dimension:
        raise SystemExit("Mismatch: query embedding length != model dimension")

    print("Embedding smoke test passed.")


if __name__ == "__main__":
    main()
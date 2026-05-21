"""
Test embedding layer on normalized knowledge base sample.

Usage:
  python scripts/test_embedding_on_sample.py \
    --config configs/rag.yaml \
    --input data/knowledge/knowledge_base_sample.jsonl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rag.document import to_embedding_documents
from src.rag.embedding import build_embedding_model_from_config
from src.rag.knowledge_loader import load_knowledge_base, validate_knowledge_base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test embedding on knowledge sample.")

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rag.yaml"),
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/knowledge/knowledge_base_sample.jsonl"),
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

    raw_docs = load_knowledge_base(args.input)
    validation = validate_knowledge_base(raw_docs)

    if validation.has_errors:
        raise SystemExit("Knowledge sample has validation errors.")

    docs = to_embedding_documents(validation.valid_docs)
    texts = [doc.text for doc in docs]

    config = load_yaml(args.config)
    embedder = build_embedding_model_from_config(config)

    embeddings = embedder.embed_texts(texts)

    print("input_docs:", len(raw_docs))
    print("valid_docs:", len(validation.valid_docs))
    print("embedding_docs:", len(docs))
    print("embeddings:", len(embeddings))
    print("embedding_dimension:", embedder.embedding_dimension)

    if len(embeddings) != len(docs):
        raise SystemExit("Mismatch: embeddings count != docs count")

    print("\nExample:")
    print("doc_id:", docs[0].doc_id)
    print("text_head:", docs[0].text[:160])
    print("metadata_keyword:", docs[0].metadata.get("keyword"))
    print("vector_head:", embeddings[0][:5])

    print("\nEmbedding on sample passed.")


if __name__ == "__main__":
    main()
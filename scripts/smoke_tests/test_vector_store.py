"""
Smoke test ChromaVectorStore with knowledge_base_sample.jsonl.

Usage:
  python scripts/test_vector_store.py \
    --config configs/rag.yaml \
    --input data/knowledge/knowledge_base_sample.jsonl \
    --recreate
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
from src.rag.vector_store import ChromaVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test Chroma vector store.")

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

    parser.add_argument(
        "--recreate",
        action="store_true",
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

    raw_docs = load_knowledge_base(args.input)
    validation = validate_knowledge_base(raw_docs)

    if validation.has_errors:
        raise SystemExit("Knowledge sample has validation errors.")

    docs = to_embedding_documents(validation.valid_docs)

    embedder = build_embedding_model_from_config(config)
    embeddings = embedder.embed_texts([doc.text for doc in docs])

    store = ChromaVectorStore.from_config_dict(config)

    if args.recreate:
        store.reset_collection()

    before_count = store.count()

    written = store.upsert_documents(
        documents=docs,
        embeddings=embeddings,
        batch_size=128,
    )

    after_count = store.count()

    print("input_docs:", len(raw_docs))
    print("valid_docs:", len(validation.valid_docs))
    print("embedding_docs:", len(docs))
    print("written:", written)
    print("before_count:", before_count)
    print("after_count:", after_count)

    query = "Bánh chưng có ý nghĩa gì trong ngày Tết?"
    query_embedding = embedder.embed_query(query)

    results = store.query(
        query_embedding,
        top_k=3,
    )

    print("\nQuery:", query)
    print("Top results:")

    for i, item in enumerate(results, start=1):
        print("-" * 80)
        print("rank:", i)
        print("doc_id:", item.doc_id)
        print("score:", item.score)
        print("distance:", item.distance)
        print("keyword:", item.metadata.get("keyword"))
        print("category:", item.metadata.get("category"))
        print("text_head:", item.text[:200])

    if after_count < len(docs):
        raise SystemExit("Chroma count is smaller than inserted docs.")

    if not results:
        raise SystemExit("Query returned no results.")

    print("\nVector store smoke test passed.")


if __name__ == "__main__":
    main()
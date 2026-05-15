"""
Smoke test document normalization.

Usage:
  python scripts/test_document_normalization.py \
    --input data/knowledge/knowledge_base_sample.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.document import to_embedding_documents
from src.rag.knowledge_loader import load_knowledge_base, validate_knowledge_base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test RAG document normalization.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/knowledge/knowledge_base_sample.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_docs = load_knowledge_base(args.input)
    validation = validate_knowledge_base(raw_docs)

    if validation.has_errors:
        raise SystemExit("Input has validation errors. Run validate_knowledge_base.py first.")

    embedding_docs = to_embedding_documents(validation.valid_docs)

    print(f"raw_docs: {len(raw_docs)}")
    print(f"valid_docs: {len(validation.valid_docs)}")
    print(f"embedding_docs: {len(embedding_docs)}")

    print("\nExample EmbeddingDocument:")
    first = embedding_docs[0]
    print(json.dumps({
        "doc_id": first.doc_id,
        "text_head": first.text[:200],
        "metadata": first.metadata,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
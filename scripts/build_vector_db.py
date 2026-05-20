"""
Build full ChromaDB vector database from knowledge_base.jsonl.

Usage:
  python scripts/build_vector_db.py \
    --config configs/rag.yaml \
    --input data/knowledge/knowledge_base.jsonl \
    --stats-output data/stats/knowledge_base_stats.json \
    --recreate
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.io import write_json
from src.rag.document import EmbeddingDocument, to_embedding_documents
from src.rag.embedding import build_embedding_model_from_config
from src.rag.knowledge_loader import (
    build_knowledge_validation_stats,
    load_knowledge_base,
    validate_knowledge_base,
)
from src.rag.vector_store import ChromaVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ChromaDB vector database from knowledge_base.jsonl."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rag.yaml"),
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Knowledge base JSONL path. Overrides config knowledge_base.input_path.",
    )

    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Stats output path. Overrides config stats.knowledge_base_stats_path.",
    )

    parser.add_argument(
        "--validation-stats-output",
        type=Path,
        default=None,
        help="Optional validation stats output path.",
    )

    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate Chroma collection before inserting.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load/validate/embed-count planning. Do not embed or write ChromaDB.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for debugging. Do not use for final build.",
    )

    parser.add_argument(
        "--upsert-batch-size",
        type=int,
        default=512,
    )

    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_input_path(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    if args.input is not None:
        return args.input

    path = config.get("knowledge_base", {}).get("input_path")
    if not path:
        raise ValueError(
            "Missing input path. Provide --input or set knowledge_base.input_path in config."
        )

    return Path(str(path))


def resolve_stats_output_path(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    if args.stats_output is not None:
        return args.stats_output

    path = config.get("stats", {}).get("knowledge_base_stats_path")
    if not path:
        return Path("data/stats/knowledge_base_stats.json")

    return Path(str(path))


def resolve_validation_stats_output_path(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> Path | None:
    if args.validation_stats_output is not None:
        return args.validation_stats_output

    path = config.get("stats", {}).get("knowledge_base_validation_stats_path")
    if not path:
        return None

    return Path(str(path))


def coerce_metadata_int(value: Any) -> int:
    if value is None:
        return 0

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        return int(value)

    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0

    return 0


def build_vector_db_stats(
    *,
    input_path: Path,
    docs: list[EmbeddingDocument],
    embedding_model_name: str,
    embedding_dimension: int | None,
    vector_count_before: int,
    vector_count_after: int,
    collection_name: str,
    persist_dir: Path,
    dry_run: bool,
    recreated: bool,
) -> dict[str, Any]:
    categories = Counter()
    keywords = Counter()
    content_lengths_words: list[int] = []
    content_lengths_chars: list[int] = []

    for doc in docs:
        categories[str(doc.metadata.get("category", ""))] += 1
        keywords[str(doc.metadata.get("keyword", ""))] += 1

        content_lengths_words.append(
            coerce_metadata_int(doc.metadata.get("content_length_words"))
        )
        content_lengths_chars.append(
            coerce_metadata_int(doc.metadata.get("content_length_chars"))
        )

    def safe_min(values: list[int]) -> int:
        return min(values) if values else 0

    def safe_max(values: list[int]) -> int:
        return max(values) if values else 0

    def safe_avg(values: list[int]) -> float:
        return round(mean(values), 2) if values else 0.0

    examples = [
        {
            "doc_id": doc.doc_id,
            "keyword": doc.metadata.get("keyword"),
            "category": doc.metadata.get("category"),
            "content_length_words": doc.metadata.get("content_length_words"),
            "text_head": doc.text[:180],
        }
        for doc in docs[:10]
    ]

    return {
        "input_path": str(input_path),
        "dry_run": dry_run,
        "recreated": recreated,
        "total_documents": len(docs),
        "category_distribution": dict(categories),
        "unique_keyword_count": len([k for k in keywords if k]),
        "top_keywords": keywords.most_common(30),
        "content_length_words": {
            "min": safe_min(content_lengths_words),
            "max": safe_max(content_lengths_words),
            "avg": safe_avg(content_lengths_words),
        },
        "content_length_chars": {
            "min": safe_min(content_lengths_chars),
            "max": safe_max(content_lengths_chars),
            "avg": safe_avg(content_lengths_chars),
        },
        "embedding_model": embedding_model_name,
        "embedding_dimension": embedding_dimension,
        "vector_db": {
            "provider": "chroma",
            "persist_dir": str(persist_dir),
            "collection_name": collection_name,
            "count_before": vector_count_before,
            "count_after": vector_count_after,
        },
        "examples": examples,
    }


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger = logging.getLogger(__name__)

    config = load_yaml(args.config)

    input_path = resolve_input_path(args, config)
    stats_output_path = resolve_stats_output_path(args, config)
    validation_stats_output_path = resolve_validation_stats_output_path(args, config)

    logger.info("Loading knowledge base from %s", input_path)
    raw_docs = load_knowledge_base(input_path)

    if args.limit is not None:
        logger.warning("Applying debug limit: %d docs", args.limit)
        raw_docs = raw_docs[: args.limit]

    logger.info("Loaded %d raw docs.", len(raw_docs))

    logger.info("Validating knowledge base.")
    validation = validate_knowledge_base(raw_docs)

    validation_stats = build_knowledge_validation_stats(raw_docs, validation)

    if validation_stats_output_path:
        write_json(validation_stats, validation_stats_output_path)
        logger.info("Validation stats written to %s", validation_stats_output_path)

    if validation.has_errors:
        raise SystemExit(
            "Knowledge base has validation errors. "
            f"See: {validation_stats_output_path}"
        )

    docs = to_embedding_documents(validation.valid_docs)

    logger.info("Normalized %d embedding documents.", len(docs))

    vector_cfg = config.get("vector_db", {})
    persist_dir = Path(str(vector_cfg.get("persist_dir", "vector_db/chroma")))
    collection_name = str(vector_cfg.get("collection_name", "cultural_knowledge"))

    embedding_cfg = config.get("embedding", {})
    embedding_model_name = str(embedding_cfg.get("model_name", ""))

    if args.dry_run:
        logger.info("Dry run enabled. Skipping embedding and ChromaDB write.")

        stats = build_vector_db_stats(
            input_path=input_path,
            docs=docs,
            embedding_model_name=embedding_model_name,
            embedding_dimension=None,
            vector_count_before=0,
            vector_count_after=0,
            collection_name=collection_name,
            persist_dir=persist_dir,
            dry_run=True,
            recreated=args.recreate,
        )

        write_json(stats, stats_output_path)
        logger.info("Dry-run stats written to %s", stats_output_path)
        return

    logger.info("Loading embedding model.")
    embedder = build_embedding_model_from_config(config)

    texts = [doc.text for doc in docs]

    logger.info("Embedding %d documents.", len(texts))
    embeddings = embedder.embed_texts(texts)

    logger.info("Opening vector store.")
    store = ChromaVectorStore.from_config_dict(config)

    if args.recreate:
        store.reset_collection()

    count_before = store.count()

    logger.info("Chroma count before upsert: %d", count_before)

    inserted = store.upsert_documents(
        documents=docs,
        embeddings=embeddings,
        batch_size=args.upsert_batch_size,
    )

    count_after = store.count()

    logger.info("Inserted/upserted documents: %d", inserted)
    logger.info("Chroma count after upsert: %d", count_after)

    stats = build_vector_db_stats(
        input_path=input_path,
        docs=docs,
        embedding_model_name=embedder.config.model_name,
        embedding_dimension=embedder.embedding_dimension,
        vector_count_before=count_before,
        vector_count_after=count_after,
        collection_name=collection_name,
        persist_dir=persist_dir,
        dry_run=False,
        recreated=args.recreate,
    )

    write_json(stats, stats_output_path)

    logger.info("Vector DB stats written to %s", stats_output_path)

    if count_after < len(docs):
        raise SystemExit(
            f"Vector DB count looks wrong: count_after={count_after}, docs={len(docs)}"
        )

    logger.info("Build vector DB completed successfully.")


if __name__ == "__main__":
    main()
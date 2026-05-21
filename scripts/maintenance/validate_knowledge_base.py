"""
Validate knowledge_base.jsonl.

Usage:
  python scripts/validate_knowledge_base.py \
    --input data/knowledge/knowledge_base.jsonl \
    --stats-output data/stats/knowledge_base_validation_stats.json \
    --valid-output data/knowledge/knowledge_base.validated.jsonl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.io import write_json, write_jsonl
from src.rag.knowledge_loader import (
    build_knowledge_validation_stats,
    load_knowledge_base,
    validate_knowledge_base,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate knowledge base JSONL.")

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/knowledge/knowledge_base.jsonl"),
    )

    parser.add_argument(
        "--stats-output",
        type=Path,
        default=Path("data/stats/knowledge_base_validation_stats.json"),
    )

    parser.add_argument(
        "--valid-output",
        type=Path,
        default=None,
        help="Optional output path for valid normalized docs.",
    )

    parser.add_argument(
        "--allow-invalid-category",
        action="store_true",
        help="Downgrade invalid category from error to warning.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger = logging.getLogger(__name__)

    logger.info("Loading knowledge base from %s", args.input)
    raw_docs = load_knowledge_base(args.input)

    logger.info("Loaded %d raw docs.", len(raw_docs))

    result = validate_knowledge_base(
        raw_docs,
        strict_category=not args.allow_invalid_category,
    )

    stats = build_knowledge_validation_stats(raw_docs, result)

    write_json(stats, args.stats_output)
    logger.info("Validation stats written to %s", args.stats_output)

    if args.valid_output:
        written = write_jsonl(result.valid_docs, args.valid_output)
        logger.info("Wrote %d valid docs to %s", written, args.valid_output)

    logger.info(
        "Validation summary — raw: %d | valid: %d | invalid: %d | errors: %d | warnings: %d",
        stats["total_raw_docs"],
        stats["valid_docs"],
        stats["invalid_docs"],
        stats["error_count"],
        stats["warning_count"],
    )

    if result.has_errors:
        raise SystemExit(
            "Knowledge base validation failed. "
            f"See report: {args.stats_output}"
        )


if __name__ == "__main__":
    main()
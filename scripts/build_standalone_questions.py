# scripts/build_standalone_questions.py
"""
CLI script: build standalone questions for a processed JSONL file.

Usage:
  python scripts/build_standalone_questions.py \
    --input data/processed/train.jsonl \
    --output data/processed/final_train.jsonl \
    --stats-output data/stats/standalone_question_stats.json \
    [--use-llm]
"""

import argparse
import logging
from pathlib import Path

from src.utils.config import get_config
from src.data.loader import load_raw_records
from src.data.io import write_jsonl, write_json
from src.standalone.standalone_service import process_records
from src.standalone.stats import compute_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build standalone questions.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stats-output", required=True, type=Path)
    parser.add_argument(
        "--use-llm", action="store_true",
        help="Enable LLM fallback (requires GEMINI_API_KEY in .env)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()
    sa_cfg = config["standalone"]

    logging.basicConfig(
        level=config["logging"]["level"],
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Loading records from {args.input}")
    records = load_raw_records(args.input)
    logger.info(f"Loaded {len(records)} records.")

    use_llm = args.use_llm or sa_cfg.get("use_llm_fallback", False)
    processed = process_records(
        records,
        use_llm_fallback=use_llm,
        llm_model=sa_cfg["llm_model"],
        llm_max_retries=sa_cfg["llm_max_retries"],
        llm_timeout=sa_cfg["llm_timeout_seconds"],
    )

    n = write_jsonl(processed, args.output)
    logger.info(f"Wrote {n} records to {args.output}")

    stats = compute_stats(records, processed)
    write_json(stats, args.stats_output)
    logger.info(f"Stats written to {args.stats_output}")
    logger.info(
        f"Summary — template: {stats['template_rewritten_count']} | "
        f"llm: {stats['llm_rewritten_count']} | "
        f"unchanged: {stats['unchanged_count']} | "
        f"failed: {stats['failed_count']}"
    )


if __name__ == "__main__":
    main()
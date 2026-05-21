"""
CLI script: optimized standalone question rewriting.

Workflow:
1. Template-only pass.
2. Export pending unchanged records.
3. Batch LLM rewrite only pending records.
4. Cache LLM outputs.
5. Merge and write final JSONL.

Usage:
  python scripts/build_standalone_questions.py \
    --input data/raw/train.jsonl \
    --output data/processed/final_train.jsonl \
    --stats-output data/stats/train_standalone_stats.json \
    --pending-output data/cache/pending_train_llm.jsonl \
    --cache-path data/cache/standalone_llm_cache.json \
    --use-llm \
    --batch-size 20 \
    --sleep-seconds 7
"""

import argparse
import logging
from pathlib import Path
import sys

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import get_config
from src.data.loader import load_raw_records, validate_image_question_ids
from src.data.io import write_jsonl, write_json
from src.standalone.batch_service import process_records_optimized
from src.standalone.stats import compute_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build standalone questions with batch LLM + cache.")

    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stats-output", required=True, type=Path)

    parser.add_argument("--pending-output", required=False, type=Path)
    parser.add_argument(
        "--cache-path",
        required=False,
        type=Path,
        default=Path("data/cache/standalone_llm_cache.json"),
    )

    parser.add_argument("--use-llm", action="store_true")

    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--sleep-seconds", type=float, default=7.0)

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
    records = validate_image_question_ids(records)
    logger.info(f"Loaded {len(records)} records.")

    use_llm = args.use_llm or sa_cfg.get("use_llm_fallback", False)

    existing_output_records = None
    if use_llm and args.output.exists():
        logger.info(f"Loading existing output checkpoint from {args.output}")
        existing_output_records = load_raw_records(args.output)

    processed = None
    interrupted = False

    try:
        processed = process_records_optimized(
            records=records,
            use_llm=use_llm,
            pending_output_path=args.pending_output,
            cache_path=args.cache_path,
            output_path=args.output,
            existing_output_records=existing_output_records,
            model=sa_cfg["llm_model"],
            batch_size=args.batch_size,
            sleep_seconds=args.sleep_seconds,
            max_retries=sa_cfg["llm_max_retries"],
            timeout=sa_cfg["llm_timeout_seconds"],
        )
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("Interrupted by user. Loading latest checkpoint for stats.")
        if args.output.exists():
            processed = load_raw_records(args.output)
        else:
            raise

    written = write_jsonl(processed, args.output)
    logger.info(f"Wrote {written} records to {args.output}")

    stats = compute_stats(records, processed)
    write_json(stats, args.stats_output)
    logger.info(f"Stats written to {args.stats_output}")

    logger.info(
        f"Summary — template: {stats['template_rewritten_count']} | "
        f"already_standalone: {stats['already_standalone_count']} | "
        f"llm: {stats['llm_rewritten_count']} | "
        f"unchanged: {stats['unchanged_count']} | "
        f"failed: {stats['failed_count']}"
    )

    if processed is not None and any(record.get("rewrite_method") == "unchanged" for record in processed):
        logger.info("Run stopped before all records were rewritten. Re-run the same command to continue.")

    if interrupted:
        sys.exit(130)


if __name__ == "__main__":
    main()

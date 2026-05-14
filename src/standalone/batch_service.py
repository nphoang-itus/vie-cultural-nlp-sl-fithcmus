"""
Optimized standalone question pipeline.

Workflow:
1. Template-only rewrite for all records.
2. Collect unchanged records.
3. Rewrite unchanged records using batched LLM calls.
4. Cache LLM outputs after each batch.
5. Merge LLM results back into final records.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from src.data.io import write_jsonl
from src.standalone.cache import (
    load_cache,
    save_cache,
    make_rewrite_cache_key,
)
from src.standalone.template_rewriter import rewrite as template_rewrite
from src.standalone.llm_rewriter import rewrite_batch_with_llm

logger = logging.getLogger(__name__)


def apply_template_only(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Apply template rewrite to all records.
    No LLM call here.
    """
    processed: list[dict[str, Any]] = []

    for i, record in enumerate(records):
        if i % 500 == 0:
            logger.info(f"Template pass: {i}/{len(records)}")

        keyword = str(record.get("keyword", "")).strip()
        question = str(record.get("question", "")).strip()

        template_result = template_rewrite(keyword, question)

        if template_result.matched:
            processed.append({
                **record,
                "standalone_question": template_result.standalone_question,
                "rewrite_method": "template",
            })
        else:
            processed.append({
                **record,
                "standalone_question": question,
                "rewrite_method": "unchanged",
            })

    return processed


def collect_pending_llm_records(
    processed_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Collect records that still need LLM rewrite.

    Criteria:
    - rewrite_method == "unchanged"
    - question is non-empty
    - vision_caption is non-empty
    """
    pending: list[dict[str, Any]] = []

    for index, record in enumerate(processed_records):
        question = str(record.get("question", "")).strip()
        vision_caption = str(record.get("vision_caption", "")).strip()

        if (
            record.get("rewrite_method") == "unchanged"
            and question
            and vision_caption
        ):
            pending.append({
                **record,
                "_record_index": index,
            })

    return pending


def apply_cached_llm_results(
    processed_records: list[dict[str, Any]],
    pending_records: list[dict[str, Any]],
    cache: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Apply cache hits before calling LLM.

    Returns remaining records that are not found in cache.
    """
    remaining: list[dict[str, Any]] = []

    cache_hits = 0

    for record in pending_records:
        vision_caption = str(record.get("vision_caption", ""))
        question = str(record.get("question", ""))
        cache_key = make_rewrite_cache_key(vision_caption, question)

        cached_item = cache.get(cache_key)

        if cached_item and cached_item.get("standalone_question"):
            record_index = int(record["_record_index"])
            processed_records[record_index] = {
                **processed_records[record_index],
                "standalone_question": cached_item["standalone_question"],
                "rewrite_method": "llm",
                "rewrite_source": "cache",
            }
            cache_hits += 1
        else:
            remaining.append(record)

    logger.info(f"Applied {cache_hits} cached LLM rewrites.")
    logger.info(f"Remaining LLM records after cache: {len(remaining)}")

    return remaining


def _batch_records(records: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    """Split records into batches."""
    return [
        records[i:i + batch_size]
        for i in range(0, len(records), batch_size)
    ]


def apply_batch_llm_rewrite(
    processed_records: list[dict[str, Any]],
    pending_records: list[dict[str, Any]],
    cache_path: Path,
    model: str,
    batch_size: int = 20,
    sleep_seconds: float = 7.0,
    max_retries: int = 2,
    timeout: int = 10,
) -> list[dict[str, Any]]:
    """
    Rewrite pending records using batched LLM calls and cache.

    Cache is saved after every successful batch.
    """
    cache = load_cache(cache_path)

    remaining_records = apply_cached_llm_results(
        processed_records=processed_records,
        pending_records=pending_records,
        cache=cache,
    )

    batches = _batch_records(remaining_records, batch_size=batch_size)

    logger.info(f"Total LLM batches to run: {len(batches)}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Sleep seconds between requests: {sleep_seconds}")

    for batch_idx, batch in enumerate(batches, start=1):
        logger.info(f"Running LLM batch {batch_idx}/{len(batches)}")

        llm_items = [
            {
                "index": int(record["_record_index"]),
                "vision_caption": str(record.get("vision_caption", "")).strip(),
                "question": str(record.get("question", "")).strip(),
            }
            for record in batch
        ]

        result = rewrite_batch_with_llm(
            items=llm_items,
            model=model,
            max_retries=max_retries,
            timeout=timeout,
        )

        if not result.success:
            logger.warning(f"Batch {batch_idx} failed. Marking records as failed.")
            for record in batch:
                record_index = int(record["_record_index"])
                processed_records[record_index] = {
                    **processed_records[record_index],
                    "rewrite_method": "failed",
                    "rewrite_source": "batch_llm_failed",
                }

            save_cache(cache, cache_path)
            time.sleep(sleep_seconds)
            continue

        success_count = 0

        for record in batch:
            record_index = int(record["_record_index"])

            standalone_question = result.results.get(record_index)

            if standalone_question:
                processed_records[record_index] = {
                    **processed_records[record_index],
                    "standalone_question": standalone_question,
                    "rewrite_method": "llm",
                    "rewrite_source": "api",
                }

                cache_key = make_rewrite_cache_key(
                    vision_caption=str(record.get("vision_caption", "")),
                    question=str(record.get("question", "")),
                )

                cache[cache_key] = {
                    "standalone_question": standalone_question,
                    "vision_caption": record.get("vision_caption", ""),
                    "question": record.get("question", ""),
                    "model": model,
                }

                success_count += 1
            else:
                processed_records[record_index] = {
                    **processed_records[record_index],
                    "rewrite_method": "failed",
                    "rewrite_source": "missing_from_batch_output",
                }

        save_cache(cache, cache_path)

        logger.info(f"Batch {batch_idx} success records: {success_count}/{len(batch)}")

        if batch_idx < len(batches):
            time.sleep(sleep_seconds)

    return processed_records


def process_records_optimized(
    records: list[dict[str, Any]],
    use_llm: bool,
    pending_output_path: Path | None,
    cache_path: Path,
    model: str,
    batch_size: int,
    sleep_seconds: float,
    max_retries: int,
    timeout: int,
) -> list[dict[str, Any]]:
    """
    Full optimized standalone-question processing pipeline.
    """
    processed_records = apply_template_only(records)

    pending_records = collect_pending_llm_records(processed_records)

    logger.info(f"Pending records for LLM: {len(pending_records)}")

    if pending_output_path:
        write_jsonl(pending_records, pending_output_path)
        logger.info(f"Wrote pending LLM records to {pending_output_path}")

    if not use_llm:
        logger.info("LLM disabled. Returning template-only results.")
        return processed_records

    final_records = apply_batch_llm_rewrite(
        processed_records=processed_records,
        pending_records=pending_records,
        cache_path=cache_path,
        model=model,
        batch_size=batch_size,
        sleep_seconds=sleep_seconds,
        max_retries=max_retries,
        timeout=timeout,
    )

    return final_records
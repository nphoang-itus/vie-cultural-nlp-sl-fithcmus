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
import unicodedata
import re

from src.data.io import write_jsonl
from src.standalone.cache import (
    load_cache,
    save_cache,
    make_rewrite_cache_key,
)
from src.standalone.template_rewriter import rewrite as template_rewrite
from src.standalone.llm_rewriter import rewrite_batch_with_llm

logger = logging.getLogger(__name__)

_VIETNAMESE_STOP_TOKENS = {
    "tet",
    "viet",
    "nam",
    "truyen",
    "thong",
    "van",
    "hoa",
}


def _strip_accents(text: str) -> str:
    """
    Convert Vietnamese text to accent-free lowercase text.

    Example:
        "bánh chưng Tết" -> "banh chung tet"
    """
    text = (text or "").lower()
    normalized = unicodedata.normalize("NFD", text)
    without_accents = "".join(
        char for char in normalized
        if unicodedata.category(char) != "Mn"
    )
    return without_accents


def _tokenize_for_subject_match(text: str) -> list[str]:
    """
    Tokenize text for rough subject matching.
    Keeps only alphanumeric Vietnamese-normalized tokens.
    """
    normalized = _strip_accents(text)
    normalized = normalized.replace("_", " ").replace("-", " ")
    return re.findall(r"[a-z0-9]+", normalized)


def _keyword_subject_tokens(keyword: str) -> list[str]:
    """
    Extract useful subject tokens from keyword.

    Example:
        "banh chung Tet" -> ["banh", "chung"]
        "ao dai" -> ["ao", "dai"]
    """
    tokens = _tokenize_for_subject_match(keyword)

    if len(tokens) <= 2:
        return tokens

    filtered = [
        token for token in tokens
        if token not in _VIETNAMESE_STOP_TOKENS
    ]

    return filtered or tokens


def _question_contains_keyword_subject(keyword: str, question: str) -> bool:
    """
    Return True if the question already contains the keyword subject.

    This is accent-insensitive:
        keyword: "banh chung Tet"
        question: "Ý nghĩa văn hóa của bánh chưng là gì?"
        -> True
    """
    subject_tokens = _keyword_subject_tokens(keyword)
    question_tokens = set(_tokenize_for_subject_match(question))

    if not subject_tokens or not question_tokens:
        return False

    matched = sum(1 for token in subject_tokens if token in question_tokens)

    # For multi-token subjects, require at least 2 matched tokens.
    # For one-token subjects, require that token.
    required = 1 if len(subject_tokens) == 1 else 2

    return matched >= required


def apply_template_only(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Apply deterministic standalone-question preprocessing.

    Stages:
    1. Template rewrite for clearly image-dependent questions.
    2. Detect questions that are already standalone.
    3. Mark the rest as unchanged so only those go to LLM.
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
                "rewrite_source": "template_rule",
            })
            continue

        if _question_contains_keyword_subject(keyword, question):
            processed.append({
                **record,
                "standalone_question": question,
                "rewrite_method": "already_standalone",
                "rewrite_source": "keyword_subject_detected",
            })
            continue

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


def apply_existing_llm_results(
    processed_records: list[dict[str, Any]],
    existing_records: list[dict[str, Any]] | None,
) -> None:
    """
    Restore successful LLM rewrites from a previous output file.

    This lets an interrupted run continue without reprocessing records that were
    already written to the final JSONL checkpoint.
    """
    if not existing_records:
        return

    existing_by_image_id = {
        str(record.get("image_id", "")).strip(): record
        for record in existing_records
        if (
            str(record.get("image_id", "")).strip()
            and record.get("rewrite_method") == "llm"
            and str(record.get("standalone_question", "")).strip()
        )
    }

    restored_count = 0

    for index, record in enumerate(processed_records):
        image_id = str(record.get("image_id", "")).strip()
        existing = existing_by_image_id.get(image_id)

        if not existing:
            continue

        if str(existing.get("question", "")).strip() != str(record.get("question", "")).strip():
            continue

        processed_records[index] = {
            **record,
            "standalone_question": existing["standalone_question"],
            "rewrite_method": "llm",
            "rewrite_source": existing.get("rewrite_source", "existing_output"),
        }
        restored_count += 1

    logger.info(f"Restored {restored_count} LLM rewrites from existing output.")


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
    checkpoint_output_path: Path | None = None,
    batch_size: int = 20,
    sleep_seconds: float = 7.0,
    max_retries: int = 2,
    timeout: int = 10,
) -> list[dict[str, Any]]:
    """
    Rewrite pending records using batched LLM calls and cache.

    Cache and checkpoint output are saved after every batch.
    """
    cache = load_cache(cache_path)

    def save_checkpoint() -> None:
        save_cache(cache, cache_path)
        if checkpoint_output_path:
            written = write_jsonl(processed_records, checkpoint_output_path)
            logger.info(f"Checkpoint wrote {written} records to {checkpoint_output_path}")

    remaining_records = apply_cached_llm_results(
        processed_records=processed_records,
        pending_records=pending_records,
        cache=cache,
    )

    save_checkpoint()

    batches = _batch_records(remaining_records, batch_size=batch_size)

    logger.info(f"Total LLM batches to run: {len(batches)}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Sleep seconds between requests: {sleep_seconds}")

    try:
        for batch_idx, batch in enumerate(batches, start=1):
            logger.info(f"Running LLM batch {batch_idx}/{len(batches)}")

            llm_items = [
                {
                    "index": int(record["_record_index"]),
                    "keyword": str(record.get("keyword", "")).strip(),
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

                save_checkpoint()
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

            save_checkpoint()

            logger.info(f"Batch {batch_idx} success records: {success_count}/{len(batch)}")

            if batch_idx < len(batches):
                time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        logger.warning("Interrupted. Saving cache and checkpoint before exit.")
        save_checkpoint()
        raise

    return processed_records


def process_records_optimized(
    records: list[dict[str, Any]],
    use_llm: bool,
    pending_output_path: Path | None,
    cache_path: Path,
    output_path: Path | None,
    existing_output_records: list[dict[str, Any]] | None,
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
    apply_existing_llm_results(processed_records, existing_output_records)

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
        checkpoint_output_path=output_path,
        batch_size=batch_size,
        sleep_seconds=sleep_seconds,
        max_retries=max_retries,
        timeout=timeout,
    )

    return final_records

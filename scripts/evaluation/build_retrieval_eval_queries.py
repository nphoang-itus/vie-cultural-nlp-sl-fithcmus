"""
Build retrieval evaluation query sets from processed VQA splits.

The RAG knowledge base is built from train, so val/test examples are filtered to
keywords that exist in the indexed KB. Train output is intended for sanity checks.

Usage:
  python scripts/evaluation/build_retrieval_eval_queries.py
"""

from __future__ import annotations

import argparse
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.io import write_jsonl
from src.data.loader import iter_jsonl


@dataclass(frozen=True)
class SplitStats:
    raw_count: int
    valid_count: int
    covered_count: int
    deduplicated_count: int
    final_count: int
    category_distribution: dict[str, int]
    keyword_coverage_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build retrieval evaluation query JSONL files from processed splits."
    )

    parser.add_argument(
        "--knowledge-base",
        type=Path,
        default=Path("data/knowledge/knowledge_base.validated.jsonl"),
        help="Validated knowledge base JSONL path.",
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/processed/final_train.jsonl"),
        help="Processed train JSONL path.",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=Path("data/processed/final_val.jsonl"),
        help="Processed validation JSONL path.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/processed/final_test.jsonl"),
        help="Processed test JSONL path.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("data/rag-evaluation/retrieval_eval_train_sanity.jsonl"),
        help="Output JSONL path for train sanity queries.",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=Path("data/rag-evaluation/retrieval_eval_val.jsonl"),
        help="Output JSONL path for validation queries.",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=Path("data/rag-evaluation/retrieval_eval_test.jsonl"),
        help="Output JSONL path for test queries.",
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=Path("data/rag-evaluation/retrieval_eval_queries.jsonl"),
        help="Output JSONL path for combined val+test queries.",
    )
    parser.add_argument(
        "--max-per-split",
        type=int,
        default=None,
        help="Optional maximum number of records to keep per split after filtering.",
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=None,
        help="Optional maximum number of records to keep per category in each split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling.",
    )

    return parser.parse_args()


def normalize_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def load_kb_keywords(path: Path) -> set[str]:
    keywords: set[str] = set()

    for row in iter_jsonl(path):
        metadata = row.get("metadata") or {}
        keyword = normalize_key(metadata.get("keyword"))
        if keyword:
            keywords.add(keyword)

    if not keywords:
        raise ValueError(f"No metadata.keyword values found in knowledge base: {path}")

    return keywords


def build_record(row: dict[str, Any], source_split: str) -> dict[str, str] | None:
    question = clean_text(row.get("question"))
    standalone_question = clean_text(row.get("standalone_question"))
    normalized_question = standalone_question or question
    keyword = clean_text(row.get("keyword"))
    category = clean_text(row.get("category"))
    subcategory = clean_text(row.get("subcategory"))

    if not question or not normalized_question or not keyword or not category:
        return None

    image_id = clean_text(row.get("image_id"))
    question_id = clean_text(row.get("question_id"))
    record_id = f"{image_id}_{question_id}" if image_id and question_id else image_id

    return {
        "id": record_id,
        "question": question,
        "normalized_question": normalized_question,
        "expected_keyword": keyword,
        "expected_category": category,
        "expected_subcategory": subcategory,
        "source_split": source_split,
    }


def deduplicate(records: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, str]] = []

    for record in records:
        key = (
            normalize_key(record["normalized_question"]),
            normalize_key(record["expected_keyword"]),
            normalize_key(record["expected_category"]),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)

    return deduped


def sample_records(
    records: list[dict[str, str]],
    *,
    max_per_split: int | None,
    max_per_category: int | None,
    rng: random.Random,
) -> list[dict[str, str]]:
    sampled = list(records)

    if max_per_category is not None:
        by_category: dict[str, list[dict[str, str]]] = defaultdict(list)
        for record in sampled:
            by_category[record["expected_category"]].append(record)

        sampled = []
        for category in sorted(by_category):
            category_records = list(by_category[category])
            rng.shuffle(category_records)
            sampled.extend(category_records[:max_per_category])

    if max_per_split is not None and len(sampled) > max_per_split:
        if max_per_category is None:
            rng.shuffle(sampled)
            sampled = sampled[:max_per_split]
        else:
            sampled = balanced_sample_by_category(sampled, max_per_split, rng)

    return sorted(sampled, key=lambda item: item["id"])


def balanced_sample_by_category(
    records: list[dict[str, str]], max_count: int, rng: random.Random
) -> list[dict[str, str]]:
    by_category: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        by_category[record["expected_category"]].append(record)

    for category_records in by_category.values():
        rng.shuffle(category_records)

    selected: list[dict[str, str]] = []
    categories = sorted(by_category)

    while len(selected) < max_count and categories:
        next_categories: list[str] = []
        for category in categories:
            category_records = by_category[category]
            if category_records and len(selected) < max_count:
                selected.append(category_records.pop())
            if category_records:
                next_categories.append(category)
        categories = next_categories

    return selected


def process_split(
    path: Path,
    *,
    source_split: str,
    kb_keywords: set[str],
    max_per_split: int | None,
    max_per_category: int | None,
    rng: random.Random,
) -> tuple[list[dict[str, str]], SplitStats]:
    raw_count = 0
    valid_records: list[dict[str, str]] = []
    covered_records: list[dict[str, str]] = []

    for row in iter_jsonl(path):
        raw_count += 1
        record = build_record(row, source_split)
        if record is None:
            continue

        valid_records.append(record)

        if normalize_key(record["expected_keyword"]) in kb_keywords:
            covered_records.append(record)

    deduped_records = deduplicate(covered_records)
    final_records = sample_records(
        deduped_records,
        max_per_split=max_per_split,
        max_per_category=max_per_category,
        rng=rng,
    )

    category_distribution = dict(
        sorted(Counter(record["expected_category"] for record in final_records).items())
    )
    valid_count = len(valid_records)
    covered_count = len(covered_records)
    keyword_coverage_rate = covered_count / valid_count if valid_count else 0.0

    stats = SplitStats(
        raw_count=raw_count,
        valid_count=valid_count,
        covered_count=covered_count,
        deduplicated_count=len(deduped_records),
        final_count=len(final_records),
        category_distribution=category_distribution,
        keyword_coverage_rate=keyword_coverage_rate,
    )

    return final_records, stats


def print_split_summary(split: str, stats: SplitStats) -> None:
    print(f"\n[{split}]")
    print(f"raw count: {stats.raw_count}")
    print(f"valid count: {stats.valid_count}")
    print(f"covered count: {stats.covered_count}")
    print(f"deduplicated count: {stats.deduplicated_count}")
    print(f"final count: {stats.final_count}")
    print(f"keyword coverage rate: {stats.keyword_coverage_rate:.2%}")
    print("category distribution:")
    if not stats.category_distribution:
        print("  <empty>")
        return
    for category, count in stats.category_distribution.items():
        print(f"  {category}: {count}")


def validate_args(args: argparse.Namespace) -> None:
    if args.max_per_split is not None and args.max_per_split <= 0:
        raise ValueError("--max-per-split must be a positive integer")
    if args.max_per_category is not None and args.max_per_category <= 0:
        raise ValueError("--max-per-category must be a positive integer")


def main() -> None:
    args = parse_args()
    validate_args(args)

    kb_keywords = load_kb_keywords(args.knowledge_base)
    print(f"Loaded {len(kb_keywords)} unique KB keywords from {args.knowledge_base}")

    split_configs = [
        ("train", args.train, args.train_output),
        ("val", args.val, args.val_output),
        ("test", args.test, args.test_output),
    ]

    outputs: dict[str, list[dict[str, str]]] = {}

    for split_index, (split, input_path, output_path) in enumerate(split_configs):
        rng = random.Random(args.seed + split_index)
        records, stats = process_split(
            input_path,
            source_split=split,
            kb_keywords=kb_keywords,
            max_per_split=args.max_per_split,
            max_per_category=args.max_per_category,
            rng=rng,
        )
        written = write_jsonl(records, output_path)
        outputs[split] = records
        print_split_summary(split, stats)
        print(f"wrote: {output_path} ({written} records)")

    combined_records = outputs["val"] + outputs["test"]
    combined_written = write_jsonl(combined_records, args.combined_output)
    print(f"\n[combined val+test]")
    print(f"final count: {combined_written}")
    print(f"wrote: {args.combined_output} ({combined_written} records)")


if __name__ == "__main__":
    main()

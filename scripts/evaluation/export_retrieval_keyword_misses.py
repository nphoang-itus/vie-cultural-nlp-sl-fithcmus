"""
Export retrieval eval rows where no retrieved result matches expected_keyword.

Usage:
  python scripts/evaluation/export_retrieval_keyword_misses.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.io import write_jsonl
from src.data.loader import iter_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export retrieval rows where hit_keyword_at_k is false."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/rag-evaluation/results/retrieval_eval_results.category_filter.jsonl"),
        help="Retrieval eval results JSONL path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/rag-evaluation/results/retrieval_eval_keyword_misses.category_filter.jsonl"
        ),
        help="Output JSONL path for keyword misses.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of wrong retrieved keywords to print per expected keyword.",
    )
    return parser.parse_args()


def slim_retrieved_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": item.get("doc_id", ""),
        "keyword": item.get("keyword", ""),
        "category": item.get("category", ""),
        "score": item.get("score"),
    }


def build_miss_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id", ""),
        "question": row.get("question", ""),
        "retrieval_query": row.get("retrieval_query", ""),
        "expected_keyword": row.get("expected_keyword", ""),
        "expected_category": row.get("expected_category", ""),
        "retrieved": [
            slim_retrieved_item(item)
            for item in row.get("retrieved", [])
            if isinstance(item, dict)
        ],
    }


def print_wrong_keyword_summary(
    grouped_counts: dict[str, Counter[str]],
    *,
    top_n: int,
) -> None:
    print("\nMost common wrongly retrieved keywords by expected_keyword")
    if not grouped_counts:
        print("<none>")
        return

    for expected_keyword in sorted(grouped_counts):
        counter = grouped_counts[expected_keyword]
        total = sum(counter.values())
        print(f"\n{expected_keyword} ({total} retrieved items)")
        for keyword, count in counter.most_common(top_n):
            printable_keyword = keyword or "<missing>"
            print(f"  {printable_keyword}: {count}")


def main() -> None:
    args = parse_args()

    if args.top_n <= 0:
        raise ValueError("--top-n must be a positive integer")

    misses: list[dict[str, Any]] = []
    grouped_counts: dict[str, Counter[str]] = defaultdict(Counter)
    total_rows = 0

    for row in iter_jsonl(args.input):
        total_rows += 1
        if row.get("hit_keyword_at_k") is not False:
            continue

        miss_row = build_miss_row(row)
        misses.append(miss_row)

        expected_keyword = str(miss_row["expected_keyword"] or "").strip()
        for item in miss_row["retrieved"]:
            keyword = str(item.get("keyword") or "").strip()
            grouped_counts[expected_keyword][keyword] += 1

    written = write_jsonl(misses, args.output)

    print(f"Input rows: {total_rows}")
    print(f"Keyword misses: {len(misses)}")
    print(f"Wrote: {args.output} ({written} rows)")
    print_wrong_keyword_summary(grouped_counts, top_n=args.top_n)


if __name__ == "__main__":
    main()

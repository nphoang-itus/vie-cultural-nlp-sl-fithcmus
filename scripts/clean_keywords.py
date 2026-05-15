"""
Clean faulty keywords in processed VQA JSONL files or knowledge_base.jsonl.

Examples:
  python scripts/clean_keywords.py \
    --mode processed \
    --input data/processed/final_train.jsonl \
    --output data/processed/final_train_clean.jsonl \
    --mapping configs/keyword_normalization.yaml \
    --report data/stats/final_train_keyword_cleaning_report.json

  python scripts/clean_keywords.py \
    --mode knowledge \
    --input data/knowledge/knowledge_base.jsonl \
    --output data/knowledge/knowledge_base_clean.jsonl \
    --mapping configs/keyword_normalization.yaml \
    --report data/stats/knowledge_base_keyword_cleaning_report.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.io import write_json, write_jsonl
from src.rag.keyword_normalizer import (
    has_vietnamese_diacritic,
    load_keyword_mapping,
    normalize_knowledge_doc,
    normalize_processed_record,
)
from src.data.loader import iter_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize faulty Vietnamese cultural keywords.")
    parser.add_argument("--mode", choices=["processed", "knowledge"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--unmapped-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping = load_keyword_mapping(args.mapping)

    normalizer: Callable[[dict[str, Any], dict[str, str]], tuple[dict[str, Any], dict[str, Any]]]
    normalizer = normalize_processed_record if args.mode == "processed" else normalize_knowledge_doc

    fixed_records: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    raw_keyword_counter: Counter[str] = Counter()
    unmapped_counter: Counter[str] = Counter()

    for record in iter_jsonl(args.input):
        raw_keyword = ""
        if args.mode == "processed":
            raw_keyword = str(record.get("keyword_raw") or record.get("keyword") or "").strip()
        else:
            metadata = record.get("metadata") or {}
            raw_keyword = str(metadata.get("keyword_raw") or metadata.get("keyword") or "").strip()

        fixed, item_report = normalizer(record, mapping)
        fixed_records.append(fixed)
        reports.append(item_report)
        raw_keyword_counter[raw_keyword] += 1

        # Heuristic: raw has no Vietnamese diacritic and was not changed by mapping.
        if raw_keyword and not has_vietnamese_diacritic(raw_keyword) and not item_report.get("changed"):
            unmapped_counter[raw_keyword] += 1

    write_jsonl(fixed_records, args.output)

    changed_count = sum(1 for item in reports if item.get("changed"))
    standalone_changed_count = sum(
        1 for item in reports
        if item.get("standalone_before") != item.get("standalone_after")
    )

    summary = {
        "mode": args.mode,
        "input": str(args.input),
        "output": str(args.output),
        "total_records": len(fixed_records),
        "changed_keyword_records": changed_count,
        "standalone_changed_records": standalone_changed_count,
        "unique_raw_keywords": len(raw_keyword_counter),
        "unmapped_suspicious_keywords_count": len(unmapped_counter),
        "unmapped_suspicious_keywords_top": unmapped_counter.most_common(100),
        "examples": reports[:20],
    }
    write_json(summary, args.report)

    if args.unmapped_output:
        args.unmapped_output.parent.mkdir(parents=True, exist_ok=True)
        with args.unmapped_output.open("w", encoding="utf-8") as f:
            for keyword, count in unmapped_counter.most_common():
                f.write(json.dumps({"keyword_raw": keyword, "count": count}, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

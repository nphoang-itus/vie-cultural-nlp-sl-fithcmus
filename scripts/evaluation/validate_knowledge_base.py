from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rag.knowledge_loader import (
    load_knowledge_base,
    validate_knowledge_base,
    build_knowledge_validation_stats,
)
from src.data.io import write_jsonl


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Vietnamese cultural knowledge_base.jsonl."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to knowledge_base.jsonl",
    )

    parser.add_argument(
        "--valid-output",
        type=Path,
        default=Path("data/knowledge/knowledge_base.validated.jsonl"),
        help="Path to write valid knowledge documents.",
    )

    parser.add_argument(
        "--stats-output",
        type=Path,
        default=Path("data/stats/knowledge_base_validation_stats.json"),
        help="Path to write validation stats.",
    )

    parser.add_argument(
        "--allow-invalid-category",
        action="store_true",
        help="Treat invalid category as warning instead of error.",
    )

    args = parser.parse_args()

    print(f"[1] Loading knowledge base: {args.input}")
    raw_docs = load_knowledge_base(args.input)
    print(f"[2] Loaded raw docs: {len(raw_docs)}")

    print("[3] Validating knowledge documents...")
    result = validate_knowledge_base(
        raw_docs,
        strict_category=not args.allow_invalid_category,
    )

    stats = build_knowledge_validation_stats(raw_docs, result)

    print("[4] Writing valid docs...")
    written = write_jsonl(result.valid_docs, args.valid_output)

    print("[5] Writing stats...")
    write_json(stats, args.stats_output)

    print()
    print("Validation summary")
    print("-" * 60)
    print(f"Total raw docs : {stats['total_raw_docs']}")
    print(f"Valid docs     : {stats['valid_docs']}")
    print(f"Invalid docs   : {stats['invalid_docs']}")
    print(f"Errors         : {stats['error_count']}")
    print(f"Warnings       : {stats['warning_count']}")
    print(f"Written docs   : {written}")
    print(f"Valid output   : {args.valid_output}")
    print(f"Stats output   : {args.stats_output}")

    if result.has_errors:
        print()
        print("[FAILED] Knowledge base has validation errors.")
        print("Open the stats JSON and fix error-level issues before building vector DB.")
        raise SystemExit(1)

    print()
    print("[OK] Knowledge base validation passed.")


if __name__ == "__main__":
    main()
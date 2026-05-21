from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rag.retriever import CulturalKnowledgeRetriever
from src.rag.keyword_normalizer import (
    load_keyword_mapping,
    normalize_keyword,
    replace_keyword_in_text,
)


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_filters(
    *,
    category: str,
    keyword: str,
    use_category_filter: bool,
    use_keyword_filter: bool,
) -> dict[str, Any] | None:
    conditions: list[dict[str, str]] = []

    if use_category_filter and category:
        conditions.append({"category": category})

    if use_keyword_filter and keyword:
        conditions.append({"keyword": keyword})

    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}


def build_output_record(
    *,
    source: dict[str, Any],
    normalized_keyword: str,
    normalized_standalone_question: str,
    retrieve_context: str,
) -> dict[str, Any]:
    """
    Keep ALL original fields from final_test.jsonl,
    then append / overwrite the normalized fields needed for consistency,
    and finally add retrieve_context.
    """
    record = dict(source)

    # Giữ toàn bộ field cũ, nhưng đảm bảo keyword / standalone_question nhất quán
    record["keyword"] = normalized_keyword
    record["standalone_question"] = normalized_standalone_question

    # Bổ sung field mới cho người A
    record["retrieve_context"] = retrieve_context

    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create final_test JSONL with all original fields preserved and one extra field: retrieve_context (top-1 RAG context)."
    )

    parser.add_argument(
        "--input",
        default="data/processed/final_test.jsonl",
        help="Input final_test JSONL path.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/final_test_retrieve_context.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--config",
        default="configs/rag.yaml",
        help="RAG config path.",
    )
    parser.add_argument(
        "--keyword-map",
        default="configs/keyword_normalization.yaml",
        help="Keyword normalization YAML path.",
    )
    parser.add_argument(
        "--use-keyword-filter",
        action="store_true",
        help="Use keyword filter during retrieval.",
    )
    parser.add_argument(
        "--no-category-filter",
        action="store_true",
        help="Disable category filter during retrieval.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N records for smoke test.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    input_path = Path(args.input)
    output_path = Path(args.output)
    config_path = Path(args.config)
    keyword_map_path = Path(args.keyword_map)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    retriever = CulturalKnowledgeRetriever.from_config_file(config_path)
    keyword_mapping = load_keyword_mapping(keyword_map_path)

    use_category_filter = not args.no_category_filter
    use_keyword_filter = bool(args.use_keyword_filter)

    rows = list(iter_jsonl(input_path))
    if args.limit is not None:
        rows = rows[: args.limit]

    output_records: list[dict[str, Any]] = []

    total = 0
    empty_query_count = 0
    empty_retrieve_count = 0

    print("Retrieval mode:")
    print("  use_category_filter:", use_category_filter)
    print("  use_keyword_filter:", use_keyword_filter)

    for line_no, source in tqdm(rows, desc="Retrieving top-1 context"):
        total += 1

        question = normalize_text(source.get("question"))
        original_standalone_question = normalize_text(source.get("standalone_question"))
        category = normalize_text(source.get("category"))

        raw_keyword = normalize_text(
            source.get("keyword_raw")
            or source.get("keyword")
        )

        normalized_keyword, _ = normalize_keyword(raw_keyword, keyword_mapping)

        normalized_standalone_question = replace_keyword_in_text(
            original_standalone_question,
            raw_keyword=raw_keyword,
            normalized_keyword=normalized_keyword,
        )

        query = normalized_standalone_question or question

        if not query:
            empty_query_count += 1
            output_records.append(
                build_output_record(
                    source=source,
                    normalized_keyword=normalized_keyword,
                    normalized_standalone_question=normalized_standalone_question,
                    retrieve_context="",
                )
            )
            continue

        filters = build_filters(
            category=category,
            keyword=normalized_keyword,
            use_category_filter=use_category_filter,
            use_keyword_filter=use_keyword_filter,
        )

        try:
            contexts = retriever.search(
                query=query,
                top_k=1,
                filters=filters,
            )
            retrieve_context = normalize_text(contexts[0].content) if contexts else ""

        except Exception as exc:
            logging.warning(
                "Retrieval failed at line %s | image_id=%s | error=%s",
                line_no,
                source.get("image_id"),
                exc,
            )
            retrieve_context = ""

        if not retrieve_context:
            empty_retrieve_count += 1

        output_records.append(
            build_output_record(
                source=source,
                normalized_keyword=normalized_keyword,
                normalized_standalone_question=normalized_standalone_question,
                retrieve_context=retrieve_context,
            )
        )

    write_jsonl(output_path, output_records)

    print("\nDone.")
    print("Input:", input_path)
    print("Output:", output_path)
    print("Total records:", total)
    print("Empty query records:", empty_query_count)
    print("Empty retrieve_context records:", empty_retrieve_count)


if __name__ == "__main__":
    main()
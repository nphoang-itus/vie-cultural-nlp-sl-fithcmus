"""
Evaluate retrieval quality using retrieval eval query JSONL files.

Usage:
  python scripts/evaluation/evaluate_retrieval.py \
    --input data/rag-evaluation/retrieval_eval_queries.jsonl \
    --output-results data/rag-evaluation/results/retrieval_eval_results.jsonl \
    --output-summary data/rag-evaluation/results/retrieval_eval_summary.json \
    --top-k 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import unicodedata
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.io import write_json, write_jsonl
from src.data.loader import iter_jsonl
from src.rag.retriever import CulturalKnowledgeRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality.")

    parser.add_argument(
        "--input",
        "--queries",
        dest="input",
        type=Path,
        default=Path("data/rag-evaluation/retrieval_eval_queries.jsonl"),
        help="Retrieval eval query JSONL path.",
    )
    parser.add_argument(
        "--output-results",
        type=Path,
        default=Path("data/rag-evaluation/results/retrieval_eval_results.jsonl"),
        help="Per-query result JSONL output path.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("data/rag-evaluation/results/retrieval_eval_summary.json"),
        help="Summary metrics JSON output path.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rag.yaml"),
        help="RAG config YAML path.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve per query.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of eval rows to process.",
    )
    parser.add_argument(
        "--use-category-filter",
        action="store_true",
        help="If enabled, pass expected_category as a metadata filter.",
    )

    return parser.parse_args()


def normalize_text(text: Any) -> str:
    """
    Normalize text for loose matching.

    This keeps Vietnamese diacritics, but normalizes unicode form,
    lowercases, and collapses whitespace.
    """
    text = str(text or "").strip().lower()
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def resolve_retrieval_query(row: dict[str, Any]) -> str:
    """
    Resolve the actual text sent to the retriever.

    New schema uses normalized_question with question fallback. The old query
    field is supported for backward compatibility.
    """
    for field in ("normalized_question", "question", "query"):
        value = str(row.get(field) or "").strip()
        if value:
            return value
    return ""


def load_eval_queries(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for idx, row in enumerate(iter_jsonl(path), start=1):
        retrieval_query = resolve_retrieval_query(row)
        expected_keyword = str(row.get("expected_keyword") or "").strip()
        expected_category = str(row.get("expected_category") or "").strip()
        expected_subcategory = str(row.get("expected_subcategory") or "").strip()

        if not retrieval_query:
            raise ValueError(f"Missing retrieval query at row {idx}")

        if not expected_keyword and not expected_category:
            raise ValueError(
                f"Row {idx} must have at least expected_keyword or expected_category"
            )

        rows.append(
            {
                **row,
                "id": str(row.get("id") or "").strip(),
                "source_split": str(row.get("source_split") or "").strip(),
                "question": str(row.get("question") or row.get("query") or "").strip(),
                "retrieval_query": retrieval_query,
                "expected_keyword": expected_keyword,
                "expected_category": expected_category,
                "expected_subcategory": expected_subcategory,
            }
        )

        if limit is not None and len(rows) >= limit:
            break

    return rows


def keyword_matches(actual: Any, expected: Any) -> bool:
    actual_norm = normalize_text(actual)
    expected_norm = normalize_text(expected)

    if not expected_norm:
        return False

    if actual_norm == expected_norm:
        return True

    # Accept child keywords, e.g. expected "áo dài" vs actual "áo dài cách tân".
    if actual_norm.startswith(expected_norm + " "):
        return True

    return False


def category_matches(actual: Any, expected: Any) -> bool:
    actual_norm = normalize_text(actual)
    expected_norm = normalize_text(expected)

    if not expected_norm:
        return False

    return actual_norm == expected_norm


def reciprocal_rank(matches: list[bool]) -> float:
    for rank, is_match in enumerate(matches, start=1):
        if is_match:
            return 1.0 / rank
    return 0.0


def evaluate_one_query(
    retriever: CulturalKnowledgeRetriever,
    row: dict[str, Any],
    *,
    top_k: int,
    use_category_filter: bool,
) -> dict[str, Any]:
    retrieval_query = row["retrieval_query"]
    expected_keyword = row.get("expected_keyword", "")
    expected_category = row.get("expected_category", "")
    expected_subcategory = row.get("expected_subcategory", "")

    filters = None
    if use_category_filter and expected_category:
        filters = {"category": expected_category}

    results = retriever.search(
        retrieval_query,
        top_k=top_k,
        filters=filters,
    )

    retrieved: list[dict[str, Any]] = []
    keyword_hits: list[bool] = []
    category_hits: list[bool] = []

    for rank, item in enumerate(results, start=1):
        actual_keyword = item.metadata.get("keyword", "")
        actual_category = item.metadata.get("category", "")
        actual_subcategory = item.metadata.get("subcategory", "")

        keyword_hit = keyword_matches(actual_keyword, expected_keyword)
        category_hit = category_matches(actual_category, expected_category)
        keyword_hits.append(keyword_hit)
        category_hits.append(category_hit)

        retrieved.append(
            {
                "rank": rank,
                "doc_id": item.doc_id,
                "keyword": actual_keyword,
                "category": actual_category,
                "subcategory": actual_subcategory,
                "score": item.score,
                "distance": item.distance,
            }
        )

    top1 = retrieved[0] if retrieved else {}

    return {
        "id": row.get("id", ""),
        "source_split": row.get("source_split", ""),
        "question": row.get("question", ""),
        "retrieval_query": retrieval_query,
        "expected_keyword": expected_keyword,
        "expected_category": expected_category,
        "expected_subcategory": expected_subcategory,
        "top1_keyword": top1.get("keyword", ""),
        "top1_category": top1.get("category", ""),
        "top1_score": top1.get("score"),
        "top1_distance": top1.get("distance"),
        "hit_keyword_at_1": bool(keyword_hits[:1] and keyword_hits[0]),
        "hit_keyword_at_k": any(keyword_hits),
        "hit_category_at_1": bool(category_hits[:1] and category_hits[0]),
        "hit_category_at_k": any(category_hits),
        "rr_keyword": reciprocal_rank(keyword_hits),
        "rr_category": reciprocal_rank(category_hits),
        "retrieved": retrieved,
    }


def safe_rate(numerator: int | float, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / denominator, 4)


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(mean(values), 6)


def summarize(results: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    total = len(results)
    no_result_count = sum(1 for item in results if not item["retrieved"])

    top1_keyword_hits = sum(1 for item in results if item["hit_keyword_at_1"])
    topk_keyword_hits = sum(1 for item in results if item["hit_keyword_at_k"])
    top1_category_hits = sum(1 for item in results if item["hit_category_at_1"])
    topk_category_hits = sum(1 for item in results if item["hit_category_at_k"])

    top1_scores = [
        item["top1_score"]
        for item in results
        if isinstance(item.get("top1_score"), int | float)
    ]
    top1_distances = [
        item["top1_distance"]
        for item in results
        if isinstance(item.get("top1_distance"), int | float)
    ]

    return {
        "total": total,
        "top_k": top_k,
        "no_result_count": no_result_count,
        "no_result_rate": safe_rate(no_result_count, total),
        "top1_keyword_accuracy": safe_rate(top1_keyword_hits, total),
        "topk_keyword_accuracy": safe_rate(topk_keyword_hits, total),
        "top1_category_accuracy": safe_rate(top1_category_hits, total),
        "topk_category_accuracy": safe_rate(topk_category_hits, total),
        "mrr_keyword": safe_rate(sum(item["rr_keyword"] for item in results), total),
        "mrr_category": safe_rate(sum(item["rr_category"] for item in results), total),
        "avg_top1_score": average(top1_scores),
        "avg_top1_distance": average(top1_distances),
    }


def print_summary(summary: dict[str, Any], *, output_results: Path, output_summary: Path) -> None:
    print("\nRetrieval evaluation summary")
    print(f"total: {summary['total']}")
    print(f"top_k: {summary['top_k']}")
    print(f"no_result_count: {summary['no_result_count']}")
    print(f"no_result_rate: {summary['no_result_rate']:.2%}")
    print(f"top1_keyword_accuracy: {summary['top1_keyword_accuracy']:.2%}")
    print(f"topk_keyword_accuracy: {summary['topk_keyword_accuracy']:.2%}")
    print(f"top1_category_accuracy: {summary['top1_category_accuracy']:.2%}")
    print(f"topk_category_accuracy: {summary['topk_category_accuracy']:.2%}")
    print(f"mrr_keyword: {summary['mrr_keyword']:.4f}")
    print(f"mrr_category: {summary['mrr_category']:.4f}")
    print(f"avg_top1_score: {summary['avg_top1_score']}")
    print(f"avg_top1_distance: {summary['avg_top1_distance']}")
    print(f"results: {output_results}")
    print(f"summary: {output_summary}")


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k <= 0:
        raise ValueError("--top-k must be a positive integer")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer")


def main() -> None:
    args = parse_args()
    validate_args(args)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger = logging.getLogger(__name__)

    logger.info("Loading eval queries from %s", args.input)
    eval_queries = load_eval_queries(args.input, limit=args.limit)
    logger.info("Loaded %d eval queries.", len(eval_queries))

    logger.info("Loading retriever from %s", args.config)
    retriever = CulturalKnowledgeRetriever.from_config_file(args.config)

    per_query_results = []

    for idx, row in enumerate(eval_queries, start=1):
        if idx == 1 or idx == len(eval_queries) or idx % 500 == 0:
            logger.info("Evaluating query %d/%d", idx, len(eval_queries))
        per_query_results.append(
            evaluate_one_query(
                retriever,
                row,
                top_k=args.top_k,
                use_category_filter=args.use_category_filter,
            )
        )

    summary = summarize(per_query_results, top_k=args.top_k)

    write_jsonl(per_query_results, args.output_results)
    write_json(summary, args.output_summary)

    print_summary(
        summary,
        output_results=args.output_results,
        output_summary=args.output_summary,
    )


if __name__ == "__main__":
    main()

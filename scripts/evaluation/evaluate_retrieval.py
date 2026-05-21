"""
Evaluate retrieval quality using retrieval_eval_queries.jsonl.

Usage:
  python scripts/evaluate_retrieval.py \
    --config configs/rag.yaml \
    --queries data/knowledge/retrieval_eval_queries.jsonl \
    --output data/stats/retrieval_smoke_test_results.json \
    --top-k 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import unicodedata
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.io import write_json
from src.data.loader import iter_jsonl
from src.rag.retriever import CulturalKnowledgeRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality.")

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rag.yaml"),
    )

    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/knowledge/retrieval_eval_queries.jsonl"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/stats/retrieval_smoke_test_results.json"),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--use-category-filter",
        action="store_true",
        help="If enabled, search with expected_category as metadata filter.",
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


def load_eval_queries(path: Path) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(path))

    valid_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        query = str(row.get("query", "")).strip()
        expected_keyword = str(row.get("expected_keyword", "")).strip()
        expected_category = str(row.get("expected_category", "")).strip()

        if not query:
            raise ValueError(f"Missing query at row {idx}")

        if not expected_keyword and not expected_category:
            raise ValueError(
                f"Row {idx} must have at least expected_keyword or expected_category"
            )

        valid_rows.append(
            {
                **row,
                "query": query,
                "expected_keyword": expected_keyword,
                "expected_category": expected_category,
            }
        )

    return valid_rows


def keyword_matches(actual: Any, expected: Any) -> bool:
    actual_norm = normalize_text(actual)
    expected_norm = normalize_text(expected)

    if not expected_norm:
        return False

    if actual_norm == expected_norm:
        return True

    # Accept child keywords, e.g.:
    # expected: "áo dài"
    # actual: "áo dài cách tân"
    if actual_norm.startswith(expected_norm + " "):
        return True

    return False


def category_matches(actual: Any, expected: Any) -> bool:
    actual_norm = normalize_text(actual)
    expected_norm = normalize_text(expected)

    if not expected_norm:
        return False

    return actual_norm == expected_norm


def evaluate_one_query(
    retriever: CulturalKnowledgeRetriever,
    row: dict[str, Any],
    *,
    top_k: int,
    use_category_filter: bool,
) -> dict[str, Any]:
    query = row["query"]
    expected_keyword = row.get("expected_keyword", "")
    expected_category = row.get("expected_category", "")

    filters = None
    if use_category_filter and expected_category:
        filters = {"category": expected_category}

    results = retriever.search(
        query,
        top_k=top_k,
        filters=filters,
    )

    result_items = []

    top1_keyword_match = False
    top1_category_match = False
    topk_keyword_match = False
    topk_category_match = False

    for rank, item in enumerate(results, start=1):
        actual_keyword = item.metadata.get("keyword", "")
        actual_category = item.metadata.get("category", "")

        kw_match = keyword_matches(actual_keyword, expected_keyword)
        cat_match = category_matches(actual_category, expected_category)

        if rank == 1:
            top1_keyword_match = kw_match
            top1_category_match = cat_match

        topk_keyword_match = topk_keyword_match or kw_match
        topk_category_match = topk_category_match or cat_match

        result_items.append(
            {
                "rank": rank,
                "doc_id": item.doc_id,
                "score": item.score,
                "distance": item.distance,
                "keyword": actual_keyword,
                "category": actual_category,
                "subcategory": item.metadata.get("subcategory", ""),
                "keyword_match": kw_match,
                "category_match": cat_match,
                "content_head": item.content[:240],
            }
        )

    return {
        "query": query,
        "expected_keyword": expected_keyword,
        "expected_category": expected_category,
        "filters": filters,
        "returned": len(results),
        "top1_keyword_match": top1_keyword_match,
        "top1_category_match": top1_category_match,
        "topk_keyword_match": topk_keyword_match,
        "topk_category_match": topk_category_match,
        "results": result_items,
    }


def safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def summarize(results: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    total = len(results)

    top1_keyword_hits = sum(1 for item in results if item["top1_keyword_match"])
    top1_category_hits = sum(1 for item in results if item["top1_category_match"])
    topk_keyword_hits = sum(1 for item in results if item["topk_keyword_match"])
    topk_category_hits = sum(1 for item in results if item["topk_category_match"])

    no_result_count = sum(1 for item in results if item["returned"] == 0)

    failures = [
        {
            "query": item["query"],
            "expected_keyword": item["expected_keyword"],
            "expected_category": item["expected_category"],
            "returned": item["returned"],
            "top_results": item["results"][:3],
        }
        for item in results
        if not item["topk_keyword_match"]
    ]

    return {
        "total_queries": total,
        "top_k": top_k,
        "no_result_count": no_result_count,

        "top1_keyword_hits": top1_keyword_hits,
        "top1_keyword_accuracy": safe_rate(top1_keyword_hits, total),

        "top1_category_hits": top1_category_hits,
        "top1_category_accuracy": safe_rate(top1_category_hits, total),

        "topk_keyword_hits": topk_keyword_hits,
        "topk_keyword_accuracy": safe_rate(topk_keyword_hits, total),

        "topk_category_hits": topk_category_hits,
        "topk_category_accuracy": safe_rate(topk_category_hits, total),

        "failure_count_by_keyword_topk": len(failures),
        "failures": failures[:30],
    }


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger = logging.getLogger(__name__)

    logger.info("Loading eval queries from %s", args.queries)
    eval_queries = load_eval_queries(args.queries)

    logger.info("Loaded %d eval queries.", len(eval_queries))

    logger.info("Loading retriever.")
    retriever = CulturalKnowledgeRetriever.from_config_file(args.config)

    per_query_results = []

    for idx, row in enumerate(eval_queries, start=1):
        logger.info("Evaluating query %d/%d", idx, len(eval_queries))

        result = evaluate_one_query(
            retriever,
            row,
            top_k=args.top_k,
            use_category_filter=args.use_category_filter,
        )

        per_query_results.append(result)

    summary = summarize(per_query_results, top_k=args.top_k)

    output = {
        "summary": summary,
        "per_query_results": per_query_results,
    }

    write_json(output, args.output)

    logger.info("Retrieval evaluation written to %s", args.output)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
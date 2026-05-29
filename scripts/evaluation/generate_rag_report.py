"""
Generate a compact RAG report from existing KB/vector/retrieval stats.

Usage:
  python scripts/generate_rag_report.py \
    --validation-stats data/stats/knowledge_base_validation_stats.json \
    --kb-stats data/stats/knowledge_base_stats.json \
    --retrieval-stats data/rag-evaluation/results/retrieval_eval_summary.json \
    --retrieval-filtered-stats data/rag-evaluation/results/retrieval_eval_summary.category_filter.json \
    --json-output data/stats/rag_report.json \
    --md-output data/stats/rag_report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RAG summary report.")

    parser.add_argument(
        "--validation-stats",
        type=Path,
        default=Path("data/stats/knowledge_base_validation_stats.json"),
    )

    parser.add_argument(
        "--kb-stats",
        type=Path,
        default=Path("data/stats/knowledge_base_stats.json"),
    )

    parser.add_argument(
        "--retrieval-stats",
        type=Path,
        default=Path("data/rag-evaluation/results/retrieval_eval_summary.json"),
    )

    parser.add_argument(
        "--retrieval-filtered-stats",
        type=Path,
        default=Path("data/rag-evaluation/results/retrieval_eval_summary.category_filter.json"),
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("data/stats/rag_report.json"),
    )

    parser.add_argument(
        "--md-output",
        type=Path,
        default=Path("data/stats/rag_report.md"),
    )

    parser.add_argument(
        "--min-topk-keyword-accuracy",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--min-topk-category-accuracy",
        type=float,
        default=0.9,
    )
    
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit with non-zero code if overall_status is FAIL.",
    )

    return parser.parse_args()


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "_missing": True,
            "_path": str(path),
        }

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_nested(data: dict[str, Any], path: list[str], default: Any = None) -> Any:
    current: Any = data

    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)

    return default if current is None else current


def get_retrieval_summary(data: dict[str, Any]) -> dict[str, Any]:
    summary = data.get("summary")
    if isinstance(summary, dict):
        return summary
    return data


def pass_fail(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def build_report(
    *,
    validation_stats: dict[str, Any],
    kb_stats: dict[str, Any],
    retrieval_stats: dict[str, Any],
    retrieval_filtered_stats: dict[str, Any],
    min_topk_keyword_accuracy: float,
    min_topk_category_accuracy: float,
) -> dict[str, Any]:
    validation_missing = validation_stats.get("_missing", False)
    kb_missing = kb_stats.get("_missing", False)
    retrieval_missing = retrieval_stats.get("_missing", False)
    retrieval_filtered_missing = retrieval_filtered_stats.get("_missing", False)

    validation = {
        "input_missing": validation_missing,
        "total_raw_docs": validation_stats.get("total_raw_docs"),
        "valid_docs": validation_stats.get("valid_docs"),
        "invalid_docs": validation_stats.get("invalid_docs"),
        "error_count": validation_stats.get("error_count"),
        "warning_count": validation_stats.get("warning_count"),
        "category_distribution": validation_stats.get("category_distribution", {}),
        "unique_keyword_count": validation_stats.get("unique_keyword_count"),
        "top_keywords": validation_stats.get("top_keywords", [])[:15],
        "issue_distribution": validation_stats.get("issue_distribution", {}),
        "status": pass_fail(
            not validation_missing
            and validation_stats.get("error_count", 1) == 0
            and validation_stats.get("invalid_docs", 1) == 0
        ),
    }

    vector_db = get_nested(kb_stats, ["vector_db"], {}) or {}

    kb_build = {
        "input_missing": kb_missing,
        "dry_run": kb_stats.get("dry_run"),
        "total_documents": kb_stats.get("total_documents"),
        "embedding_model": kb_stats.get("embedding_model"),
        "embedding_dimension": kb_stats.get("embedding_dimension"),
        "vector_db": vector_db,
        "category_distribution": kb_stats.get("category_distribution", {}),
        "unique_keyword_count": kb_stats.get("unique_keyword_count"),
        "content_length_words": kb_stats.get("content_length_words", {}),
        "status": pass_fail(
            not kb_missing
            and kb_stats.get("dry_run") is False
            and kb_stats.get("total_documents", 0) > 0
            and vector_db.get("count_after") == kb_stats.get("total_documents")
        ),
    }

    retrieval_summary = get_retrieval_summary(retrieval_stats)

    retrieval = {
        "input_missing": retrieval_missing,
        "total_queries": retrieval_summary.get("total_queries", retrieval_summary.get("total")),
        "top_k": retrieval_summary.get("top_k"),
        "no_result_count": retrieval_summary.get("no_result_count"),
        "top1_keyword_accuracy": retrieval_summary.get("top1_keyword_accuracy"),
        "top1_category_accuracy": retrieval_summary.get("top1_category_accuracy"),
        "topk_keyword_accuracy": retrieval_summary.get("topk_keyword_accuracy"),
        "topk_category_accuracy": retrieval_summary.get("topk_category_accuracy"),
        "failure_count_by_keyword_topk": retrieval_summary.get("failure_count_by_keyword_topk"),
        "failures": retrieval_summary.get("failures", [])[:10],
        "status": pass_fail(
            not retrieval_missing
            and retrieval_summary.get("no_result_count", 1) == 0
            and float(retrieval_summary.get("topk_keyword_accuracy", 0.0)) >= min_topk_keyword_accuracy
            and float(retrieval_summary.get("topk_category_accuracy", 0.0)) >= min_topk_category_accuracy
        ),
    }

    filtered_summary = get_retrieval_summary(retrieval_filtered_stats)

    retrieval_with_filter = {
        "input_missing": retrieval_filtered_missing,
        "total_queries": filtered_summary.get("total_queries", filtered_summary.get("total")),
        "top_k": filtered_summary.get("top_k"),
        "no_result_count": filtered_summary.get("no_result_count"),
        "top1_keyword_accuracy": filtered_summary.get("top1_keyword_accuracy"),
        "top1_category_accuracy": filtered_summary.get("top1_category_accuracy"),
        "topk_keyword_accuracy": filtered_summary.get("topk_keyword_accuracy"),
        "topk_category_accuracy": filtered_summary.get("topk_category_accuracy"),
        "failure_count_by_keyword_topk": filtered_summary.get("failure_count_by_keyword_topk"),
        "failures": filtered_summary.get("failures", [])[:10],
        "status": pass_fail(
            not retrieval_filtered_missing
            and filtered_summary.get("no_result_count", 1) == 0
            and float(filtered_summary.get("topk_keyword_accuracy", 0.0)) >= min_topk_keyword_accuracy
            and float(filtered_summary.get("topk_category_accuracy", 0.0)) >= min_topk_category_accuracy
        ),
    }

    overall_pass = all(
        section["status"] == "PASS"
        for section in [
            validation,
            kb_build,
            retrieval,
        ]
    )

    recommendations: list[str] = []

    if validation["status"] != "PASS":
        recommendations.append(
            "Fix knowledge_base validation errors before rebuilding Vector DB."
        )

    if validation.get("warning_count", 0) and validation.get("warning_count", 0) > 0:
        recommendations.append(
            "Review validation warnings. Warnings may not block RAG, but they can hide metadata quality issues."
        )

    if kb_build["status"] != "PASS":
        recommendations.append(
            "Rebuild full Vector DB and ensure vector_db.count_after equals total_documents."
        )

    if retrieval["status"] != "PASS":
        recommendations.append(
            "Improve retrieval quality: check eval expected_keyword, keyword normalization, parent/child keywords, and consider top_k=5."
        )

    top1_keyword_accuracy = retrieval.get("top1_keyword_accuracy")
    topk_keyword_accuracy = retrieval.get("topk_keyword_accuracy")

    if (
        isinstance(top1_keyword_accuracy, (int, float))
        and isinstance(topk_keyword_accuracy, (int, float))
        and top1_keyword_accuracy < topk_keyword_accuracy
    ):
        recommendations.append(
            "Top-k keyword accuracy is higher than top-1. Use top_k=3 or top_k=5 in inference instead of relying only on rank 1."
        )

    return {
        "overall_status": pass_fail(overall_pass),
        "thresholds": {
            "min_topk_keyword_accuracy": min_topk_keyword_accuracy,
            "min_topk_category_accuracy": min_topk_category_accuracy,
        },
        "knowledge_base_validation": validation,
        "knowledge_base_build": kb_build,
        "retrieval": retrieval,
        "retrieval_with_category_filter": retrieval_with_filter,
        "recommendations": recommendations,
    }


def format_percent(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value * 100:.2f}%"
    return "N/A"


def render_markdown(report: dict[str, Any]) -> str:
    validation = report["knowledge_base_validation"]
    kb_build = report["knowledge_base_build"]
    retrieval = report["retrieval"]
    retrieval_with_filter = report["retrieval_with_category_filter"]

    lines: list[str] = []

    lines.append("# RAG Report")
    lines.append("")
    lines.append(f"**Overall status:** `{report['overall_status']}`")
    lines.append("")

    lines.append("## 1. Knowledge Base Validation")
    lines.append("")
    lines.append(f"- Status: `{validation['status']}`")
    lines.append(f"- Total raw docs: `{validation.get('total_raw_docs')}`")
    lines.append(f"- Valid docs: `{validation.get('valid_docs')}`")
    lines.append(f"- Invalid docs: `{validation.get('invalid_docs')}`")
    lines.append(f"- Error count: `{validation.get('error_count')}`")
    lines.append(f"- Warning count: `{validation.get('warning_count')}`")
    lines.append(f"- Unique keywords: `{validation.get('unique_keyword_count')}`")
    lines.append("")

    lines.append("### Category distribution")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("|---|---:|")
    for category, count in validation.get("category_distribution", {}).items():
        lines.append(f"| `{category}` | {count} |")
    lines.append("")

    lines.append("### Issue distribution")
    lines.append("")
    issue_distribution = validation.get("issue_distribution", {})
    if issue_distribution:
        lines.append("| Issue | Count |")
        lines.append("|---|---:|")
        for issue, count in issue_distribution.items():
            lines.append(f"| `{issue}` | {count} |")
    else:
        lines.append("No validation issues.")
    lines.append("")

    lines.append("## 2. Knowledge Base Vector Build")
    lines.append("")
    lines.append(f"- Status: `{kb_build['status']}`")
    lines.append(f"- Dry run: `{kb_build.get('dry_run')}`")
    lines.append(f"- Total documents: `{kb_build.get('total_documents')}`")
    lines.append(f"- Embedding model: `{kb_build.get('embedding_model')}`")
    lines.append(f"- Embedding dimension: `{kb_build.get('embedding_dimension')}`")

    vector_db = kb_build.get("vector_db", {}) or {}
    lines.append(f"- Chroma collection: `{vector_db.get('collection_name')}`")
    lines.append(f"- Persist dir: `{vector_db.get('persist_dir')}`")
    lines.append(f"- Count before: `{vector_db.get('count_before')}`")
    lines.append(f"- Count after: `{vector_db.get('count_after')}`")
    lines.append("")

    content_words = kb_build.get("content_length_words", {}) or {}
    lines.append("### Content length words")
    lines.append("")
    lines.append(f"- Min: `{content_words.get('min')}`")
    lines.append(f"- Avg: `{content_words.get('avg')}`")
    lines.append(f"- Max: `{content_words.get('max')}`")
    lines.append("")

    lines.append("## 3. Retrieval Evaluation")
    lines.append("")
    lines.append(f"- Status: `{retrieval['status']}`")
    lines.append(f"- Total queries: `{retrieval.get('total_queries')}`")
    lines.append(f"- Top-k: `{retrieval.get('top_k')}`")
    lines.append(f"- No-result count: `{retrieval.get('no_result_count')}`")
    lines.append(f"- Top-1 keyword accuracy: `{format_percent(retrieval.get('top1_keyword_accuracy'))}`")
    lines.append(f"- Top-1 category accuracy: `{format_percent(retrieval.get('top1_category_accuracy'))}`")
    lines.append(f"- Top-k keyword accuracy: `{format_percent(retrieval.get('topk_keyword_accuracy'))}`")
    lines.append(f"- Top-k category accuracy: `{format_percent(retrieval.get('topk_category_accuracy'))}`")
    lines.append("")

    failures = retrieval.get("failures", [])
    lines.append("### Retrieval failures")
    lines.append("")
    if failures:
        for idx, failure in enumerate(failures, start=1):
            lines.append(f"#### Failure {idx}")
            lines.append("")
            lines.append(f"- Query: `{failure.get('query')}`")
            lines.append(f"- Expected keyword: `{failure.get('expected_keyword')}`")
            lines.append(f"- Expected category: `{failure.get('expected_category')}`")
            lines.append("- Top results:")

            for item in failure.get("top_results", []):
                lines.append(
                    f"  - Rank {item.get('rank')}: "
                    f"`{item.get('keyword')}` / `{item.get('category')}` "
                    f"(score={item.get('score')}, distance={item.get('distance')})"
                )
            lines.append("")
    else:
        lines.append("No top-k keyword failures.")
        lines.append("")

    lines.append("## 4. Retrieval Evaluation With Category Filter")
    lines.append("")
    lines.append(f"- Status: `{retrieval_with_filter['status']}`")
    lines.append(f"- Total queries: `{retrieval_with_filter.get('total_queries')}`")
    lines.append(f"- Top-k: `{retrieval_with_filter.get('top_k')}`")
    lines.append(f"- No-result count: `{retrieval_with_filter.get('no_result_count')}`")
    lines.append(f"- Top-1 keyword accuracy: `{format_percent(retrieval_with_filter.get('top1_keyword_accuracy'))}`")
    lines.append(f"- Top-1 category accuracy: `{format_percent(retrieval_with_filter.get('top1_category_accuracy'))}`")
    lines.append(f"- Top-k keyword accuracy: `{format_percent(retrieval_with_filter.get('topk_keyword_accuracy'))}`")
    lines.append(f"- Top-k category accuracy: `{format_percent(retrieval_with_filter.get('topk_category_accuracy'))}`")
    lines.append("")

    lines.append("## 5. Recommendations")
    lines.append("")
    recommendations = report.get("recommendations", [])
    if recommendations:
        for item in recommendations:
            lines.append(f"- {item}")
    else:
        lines.append("- No major action required.")
    lines.append("")

    return "\n".join(lines)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()

    validation_stats = load_json_if_exists(args.validation_stats)
    kb_stats = load_json_if_exists(args.kb_stats)
    retrieval_stats = load_json_if_exists(args.retrieval_stats)
    retrieval_filtered_stats = load_json_if_exists(args.retrieval_filtered_stats)

    report = build_report(
        validation_stats=validation_stats,
        kb_stats=kb_stats,
        retrieval_stats=retrieval_stats,
        retrieval_filtered_stats=retrieval_filtered_stats,
        min_topk_keyword_accuracy=args.min_topk_keyword_accuracy,
        min_topk_category_accuracy=args.min_topk_category_accuracy,
    )

    write_json(report, args.json_output)
    write_text(args.md_output, render_markdown(report))

    print(json.dumps({
        "overall_status": report["overall_status"],
        "json_output": str(args.json_output),
        "md_output": str(args.md_output),
        "retrieval_status": report["retrieval"]["status"],
        "kb_build_status": report["knowledge_base_build"]["status"],
        "validation_status": report["knowledge_base_validation"]["status"],
    }, ensure_ascii=False, indent=2))

    
    if args.fail_on_error and report["overall_status"] != "PASS":
        raise SystemExit("RAG report failed.")

if __name__ == "__main__":
    main()

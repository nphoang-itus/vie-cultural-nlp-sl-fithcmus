"""
Build a compact error analysis report from retrieval keyword misses.

Usage:
  python scripts/evaluation/build_retrieval_error_analysis.py
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.io import write_jsonl
from src.data.loader import iter_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build representative retrieval keyword-miss error analysis."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/rag-evaluation/results/retrieval_eval_keyword_misses.category_filter.top10.jsonl"
        ),
        help="Keyword misses JSONL path.",
    )
    parser.add_argument(
        "--representatives-output",
        type=Path,
        default=Path(
            "data/rag-evaluation/results/retrieval_eval_keyword_misses.category_filter.top10.representative.jsonl"
        ),
        help="Output JSONL path for representative cases.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path(
            "data/rag-evaluation/results/retrieval_eval_keyword_misses.category_filter.top10.error_analysis.md"
        ),
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=30,
        help="Number of representative cases to export.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of aggregate rows to include in report tables.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def top_retrieved(row: dict[str, Any]) -> dict[str, Any]:
    retrieved = row.get("retrieved") or []
    if not retrieved:
        return {}
    first = retrieved[0]
    return first if isinstance(first, dict) else {}


def retrieved_keywords(row: dict[str, Any]) -> list[str]:
    keywords: list[str] = []
    for item in row.get("retrieved", []):
        if isinstance(item, dict):
            keyword = str(item.get("keyword") or "").strip()
            if keyword:
                keywords.append(keyword)
    return keywords


def infer_error_type(row: dict[str, Any]) -> str:
    expected = normalize_text(row.get("expected_keyword"))
    top_keyword = normalize_text(top_retrieved(row).get("keyword"))

    if not top_keyword:
        return "no_retrieved_keyword"

    expected_tokens = set(expected.split())
    top_tokens = set(top_keyword.split())
    overlap = expected_tokens & top_tokens

    if expected and top_keyword and (expected in top_keyword or top_keyword in expected):
        return "keyword_variant_or_parent_child"
    if overlap:
        return "same_keyword_family"

    expected_prefix = expected.split()[0] if expected else ""
    top_prefix = top_keyword.split()[0] if top_keyword else ""
    if expected_prefix and expected_prefix == top_prefix:
        return "same_domain_prefix"

    return "semantic_neighbor"


def summarize_case(row: dict[str, Any]) -> dict[str, Any]:
    top = top_retrieved(row)
    return {
        "id": row.get("id", ""),
        "question": row.get("question", ""),
        "retrieval_query": row.get("retrieval_query", ""),
        "expected_keyword": row.get("expected_keyword", ""),
        "expected_category": row.get("expected_category", ""),
        "top_retrieved_keyword": top.get("keyword", ""),
        "top_retrieved_doc_id": top.get("doc_id", ""),
        "top_retrieved_score": top.get("score"),
        "error_type": infer_error_type(row),
        "retrieved": row.get("retrieved", []),
    }


def select_representative_cases(
    rows: list[dict[str, Any]],
    *,
    num_cases: int,
) -> list[dict[str, Any]]:
    by_expected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_expected[str(row.get("expected_keyword") or "")].append(row)

    expected_order = [
        keyword
        for keyword, _ in Counter(
            str(row.get("expected_keyword") or "") for row in rows
        ).most_common()
    ]

    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    # First pass: cover as many high-volume expected keywords as possible.
    for expected_keyword in expected_order:
        candidates = sorted(
            by_expected[expected_keyword],
            key=lambda row: float(top_retrieved(row).get("score") or 0.0),
            reverse=True,
        )
        if not candidates:
            continue
        case = candidates[0]
        case_id = str(case.get("id") or "")
        if case_id not in seen_ids:
            selected.append(case)
            seen_ids.add(case_id)
        if len(selected) >= num_cases:
            break

    # Second pass: if needed, fill with the highest-scoring remaining misses.
    if len(selected) < num_cases:
        remaining = sorted(
            rows,
            key=lambda row: float(top_retrieved(row).get("score") or 0.0),
            reverse=True,
        )
        for case in remaining:
            case_id = str(case.get("id") or "")
            if case_id in seen_ids:
                continue
            selected.append(case)
            seen_ids.add(case_id)
            if len(selected) >= num_cases:
                break

    return [summarize_case(row) for row in selected]


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        clean_row = [str(value).replace("\n", " ").replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(clean_row) + " |")
    return "\n".join(lines)


def build_report(
    rows: list[dict[str, Any]],
    representatives: list[dict[str, Any]],
    *,
    top_n: int,
) -> str:
    expected_counter = Counter(str(row.get("expected_keyword") or "") for row in rows)
    category_counter = Counter(str(row.get("expected_category") or "") for row in rows)
    error_type_counter = Counter(infer_error_type(row) for row in rows)

    wrong_pair_counter: Counter[tuple[str, str]] = Counter()
    grouped_wrong_keywords: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        expected = str(row.get("expected_keyword") or "")
        for keyword in retrieved_keywords(row):
            wrong_pair_counter[(expected, keyword)] += 1
            grouped_wrong_keywords[expected][keyword] += 1

    top_expected_rows = [
        [
            keyword,
            count,
            ", ".join(
                f"{wrong_keyword} ({wrong_count})"
                for wrong_keyword, wrong_count in grouped_wrong_keywords[keyword].most_common(3)
            ),
        ]
        for keyword, count in expected_counter.most_common(top_n)
    ]
    top_pair_rows = [
        [expected, wrong, count]
        for (expected, wrong), count in wrong_pair_counter.most_common(top_n)
    ]
    representative_rows = [
        [
            idx,
            row["id"],
            row["expected_keyword"],
            row["expected_category"],
            row["top_retrieved_keyword"],
            f"{row['top_retrieved_score']:.4f}"
            if isinstance(row.get("top_retrieved_score"), int | float)
            else "",
            row["error_type"],
            row["retrieval_query"],
        ]
        for idx, row in enumerate(representatives, start=1)
    ]

    sections = [
        "# Retrieval Keyword Miss Error Analysis",
        "",
        "Source: `retrieval_eval_results.category_filter.top10.jsonl` misses.",
        "",
        f"- Keyword miss rows: {len(rows)}",
        f"- Representative cases exported: {len(representatives)}",
        "",
        "## Misses By Category",
        "",
        markdown_table(
            ["expected_category", "miss_count"],
            [[category, count] for category, count in category_counter.most_common()],
        ),
        "",
        "## Error Type Heuristic",
        "",
        markdown_table(
            ["error_type", "miss_count"],
            [[error_type, count] for error_type, count in error_type_counter.most_common()],
        ),
        "",
        "## Top Expected Keywords With Misses",
        "",
        markdown_table(
            ["expected_keyword", "miss_count", "most_common_wrong_retrieved_keywords"],
            top_expected_rows,
        ),
        "",
        "## Top Wrong Keyword Pairs",
        "",
        markdown_table(["expected_keyword", "wrong_retrieved_keyword", "count"], top_pair_rows),
        "",
        "## Representative Cases",
        "",
        markdown_table(
            [
                "#",
                "id",
                "expected_keyword",
                "category",
                "top_wrong_keyword",
                "top_score",
                "error_type",
                "retrieval_query",
            ],
            representative_rows,
        ),
        "",
        "## Notes",
        "",
        "- The top-10 category filter removes cross-category errors; remaining misses are mostly keyword-level confusions inside the expected category.",
        "- High-frequency clusters include visually or semantically close families: architecture landmarks, festivals, cakes/foods, traditional clothing, instruments, and sports.",
        "- Cases where the wrong top keyword is a parent/child or sibling concept may need keyword aliasing, richer query terms, or reranking with expected keyword candidates.",
        "",
    ]
    return "\n".join(sections)


def main() -> None:
    args = parse_args()
    if args.num_cases <= 0:
        raise ValueError("--num-cases must be a positive integer")
    if args.top_n <= 0:
        raise ValueError("--top-n must be a positive integer")

    rows = list(iter_jsonl(args.input))
    representatives = select_representative_cases(rows, num_cases=args.num_cases)

    written = write_jsonl(representatives, args.representatives_output)

    report = build_report(rows, representatives, top_n=args.top_n)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(report, encoding="utf-8")

    print(f"Input miss rows: {len(rows)}")
    print(f"Representative cases: {written}")
    print(f"Wrote: {args.representatives_output}")
    print(f"Wrote: {args.report_output}")


if __name__ == "__main__":
    main()

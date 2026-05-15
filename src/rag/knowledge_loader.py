"""
Knowledge base loading and validation utilities.

This module is responsible only for:
- reading knowledge_base.jsonl
- validating the knowledge document contract
- producing validation stats

It does not create embeddings and does not talk to ChromaDB.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from src.data.loader import iter_jsonl


ALLOWED_CATEGORIES: set[str] = {
    "kien_truc",
    "am_thuc",
    "trang_phuc",
    "le_hoi",
    "nhac_cu",
    "the_thao_truyen_thong",
}


class KnowledgeMetadata(TypedDict, total=False):
    keyword: str
    keyword_raw: str
    keyword_slug: str
    category: str
    subcategory: str
    base_image_id: str
    source_image_ids: list[str]
    source_count: int


class KnowledgeDocument(TypedDict):
    schema_version: str
    doc_id: str
    content: str
    metadata: KnowledgeMetadata


@dataclass
class ValidationIssue:
    level: str  # "error" | "warning"
    doc_id: str | None
    row_index: int
    field: str
    message: str


@dataclass
class KnowledgeValidationResult:
    valid_docs: list[KnowledgeDocument]
    issues: list[ValidationIssue]

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.level == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.level == "warning")

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0


def load_knowledge_base(path: Path) -> list[dict[str, Any]]:
    """
    Load knowledge base JSONL as raw dicts.

    Validation is intentionally separate so callers can inspect bad rows.
    """
    return list(iter_jsonl(path))


def _as_non_empty_str(value: Any) -> str:
    return str(value or "").strip()


def _add_issue(
    issues: list[ValidationIssue],
    *,
    level: str,
    row_index: int,
    doc_id: str | None,
    field: str,
    message: str,
) -> None:
    issues.append(
        ValidationIssue(
            level=level,
            row_index=row_index,
            doc_id=doc_id,
            field=field,
            message=message,
        )
    )


def validate_knowledge_doc(
    raw_doc: dict[str, Any],
    *,
    row_index: int,
    seen_doc_ids: set[str],
    strict_category: bool = True,
) -> tuple[KnowledgeDocument | None, list[ValidationIssue]]:
    """
    Validate one knowledge document.

    Error-level issues make the document invalid.
    Warning-level issues keep the document valid.
    """
    issues: list[ValidationIssue] = []

    doc_id = _as_non_empty_str(raw_doc.get("doc_id"))
    content = _as_non_empty_str(raw_doc.get("content"))
    schema_version = _as_non_empty_str(raw_doc.get("schema_version")) or "1.0"
    metadata = raw_doc.get("metadata")

    if not doc_id:
        _add_issue(
            issues,
            level="error",
            row_index=row_index,
            doc_id=None,
            field="doc_id",
            message="Missing or empty doc_id.",
        )

    if doc_id and doc_id in seen_doc_ids:
        _add_issue(
            issues,
            level="error",
            row_index=row_index,
            doc_id=doc_id,
            field="doc_id",
            message=f"Duplicate doc_id: {doc_id!r}.",
        )

    if not content:
        _add_issue(
            issues,
            level="error",
            row_index=row_index,
            doc_id=doc_id or None,
            field="content",
            message="Missing or empty content.",
        )

    if content and len(content) < 20:
        _add_issue(
            issues,
            level="warning",
            row_index=row_index,
            doc_id=doc_id or None,
            field="content",
            message="Content is very short, retrieval quality may be weak.",
        )

    if not isinstance(metadata, dict):
        _add_issue(
            issues,
            level="error",
            row_index=row_index,
            doc_id=doc_id or None,
            field="metadata",
            message="Missing metadata object.",
        )
        metadata = {}

    keyword = _as_non_empty_str(metadata.get("keyword"))
    category = _as_non_empty_str(metadata.get("category"))
    subcategory = _as_non_empty_str(metadata.get("subcategory"))
    base_image_id = _as_non_empty_str(metadata.get("base_image_id"))
    source_image_ids = metadata.get("source_image_ids")
    source_count = metadata.get("source_count")

    if not keyword:
        _add_issue(
            issues,
            level="error",
            row_index=row_index,
            doc_id=doc_id or None,
            field="metadata.keyword",
            message="Missing or empty metadata.keyword.",
        )

    if not category:
        _add_issue(
            issues,
            level="error",
            row_index=row_index,
            doc_id=doc_id or None,
            field="metadata.category",
            message="Missing or empty metadata.category.",
        )
    elif category not in ALLOWED_CATEGORIES:
        _add_issue(
            issues,
            level="error" if strict_category else "warning",
            row_index=row_index,
            doc_id=doc_id or None,
            field="metadata.category",
            message=f"Invalid category {category!r}.",
        )

    if not subcategory:
        _add_issue(
            issues,
            level="warning",
            row_index=row_index,
            doc_id=doc_id or None,
            field="metadata.subcategory",
            message="Missing metadata.subcategory.",
        )

    if not base_image_id:
        _add_issue(
            issues,
            level="warning",
            row_index=row_index,
            doc_id=doc_id or None,
            field="metadata.base_image_id",
            message="Missing metadata.base_image_id.",
        )

    if not isinstance(source_image_ids, list):
        _add_issue(
            issues,
            level="warning",
            row_index=row_index,
            doc_id=doc_id or None,
            field="metadata.source_image_ids",
            message="metadata.source_image_ids should be list[str].",
        )
        normalized_source_image_ids: list[str] = []
    else:
        normalized_source_image_ids = [
            _as_non_empty_str(item)
            for item in source_image_ids
            if _as_non_empty_str(item)
        ]

        if not normalized_source_image_ids:
            _add_issue(
                issues,
                level="warning",
                row_index=row_index,
                doc_id=doc_id or None,
                field="metadata.source_image_ids",
                message="metadata.source_image_ids is empty.",
            )

    if not isinstance(source_count, int):
        _add_issue(
            issues,
            level="warning",
            row_index=row_index,
            doc_id=doc_id or None,
            field="metadata.source_count",
            message="metadata.source_count should be an integer.",
        )
        normalized_source_count = len(normalized_source_image_ids)
    else:
        normalized_source_count = source_count

    if (
        normalized_source_image_ids
        and normalized_source_count != len(normalized_source_image_ids)
    ):
        _add_issue(
            issues,
            level="warning",
            row_index=row_index,
            doc_id=doc_id or None,
            field="metadata.source_count",
            message=(
                "metadata.source_count does not match "
                "len(metadata.source_image_ids)."
            ),
        )

    if keyword and content and keyword.lower() not in content.lower():
        _add_issue(
            issues,
            level="warning",
            row_index=row_index,
            doc_id=doc_id or None,
            field="content",
            message="metadata.keyword does not appear in content.",
        )

    # doc_id format sanity check
    if doc_id:
        parts = doc_id.split("|")
        if len(parts) < 3:
            _add_issue(
                issues,
                level="warning",
                row_index=row_index,
                doc_id=doc_id,
                field="doc_id",
                message="doc_id should follow <category>|<subcategory>|<hash>.",
            )
        else:
            doc_category = parts[0]
            doc_subcategory = parts[1]

            if category and doc_category != category:
                _add_issue(
                    issues,
                    level="warning",
                    row_index=row_index,
                    doc_id=doc_id,
                    field="doc_id",
                    message="doc_id category does not match metadata.category.",
                )

            if subcategory and doc_subcategory != subcategory:
                _add_issue(
                    issues,
                    level="warning",
                    row_index=row_index,
                    doc_id=doc_id,
                    field="doc_id",
                    message="doc_id subcategory does not match metadata.subcategory.",
                )

    has_error = any(issue.level == "error" for issue in issues)

    if has_error:
        return None, issues

    if doc_id:
        seen_doc_ids.add(doc_id)

    metadata_dict: KnowledgeMetadata = {
        "keyword": keyword,
        "category": category,
        "subcategory": subcategory,
        "base_image_id": base_image_id,
        "source_image_ids": normalized_source_image_ids,
        "source_count": normalized_source_count,
    }
    
    normalized_doc: KnowledgeDocument = {
        "schema_version": schema_version,
        "doc_id": doc_id,
        "content": content,
        "metadata": metadata_dict,
    }

    return normalized_doc, issues


def validate_knowledge_base(
    raw_docs: list[dict[str, Any]],
    *,
    strict_category: bool = True,
) -> KnowledgeValidationResult:
    """
    Validate all knowledge documents and return valid docs + issues.
    """
    seen_doc_ids: set[str] = set()
    valid_docs: list[KnowledgeDocument] = []
    all_issues: list[ValidationIssue] = []

    for row_index, raw_doc in enumerate(raw_docs):
        doc, issues = validate_knowledge_doc(
            raw_doc,
            row_index=row_index,
            seen_doc_ids=seen_doc_ids,
            strict_category=strict_category,
        )

        all_issues.extend(issues)

        if doc is not None:
            valid_docs.append(doc)

    return KnowledgeValidationResult(
        valid_docs=valid_docs,
        issues=all_issues,
    )


def build_knowledge_validation_stats(
    raw_docs: list[dict[str, Any]],
    result: KnowledgeValidationResult,
) -> dict[str, Any]:
    """
    Build a compact stats dict for JSON report.
    """
    categories = Counter()
    keywords = Counter()

    for doc in result.valid_docs:
        metadata = doc["metadata"]
        categories[metadata.get("category", "")] += 1
        keywords[metadata.get("keyword", "")] += 1

    issue_counter = Counter(
        f"{issue.level}:{issue.field}"
        for issue in result.issues
    )

    examples = [
        {
            "level": issue.level,
            "row_index": issue.row_index,
            "doc_id": issue.doc_id,
            "field": issue.field,
            "message": issue.message,
        }
        for issue in result.issues[:30]
    ]

    return {
        "total_raw_docs": len(raw_docs),
        "valid_docs": len(result.valid_docs),
        "invalid_docs": len(raw_docs) - len(result.valid_docs),
        "error_count": result.error_count,
        "warning_count": result.warning_count,
        "category_distribution": dict(categories),
        "unique_keyword_count": len([k for k in keywords if k]),
        "top_keywords": keywords.most_common(30),
        "issue_distribution": dict(issue_counter),
        "issue_examples": examples,
    }
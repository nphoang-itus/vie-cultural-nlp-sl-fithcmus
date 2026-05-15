"""
Keyword normalization utilities for Vietnamese Cultural VQA.

Purpose:
- Preserve raw dataset keyword as keyword_raw.
- Replace official keyword with normalized Vietnamese keyword.
- Generate stable keyword_slug for IDs/metadata.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import yaml


_VIETNAMESE_DIACRITIC_RE = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵ"
    r"èéẹẻẽêềếệểễ"
    r"ìíịỉĩ"
    r"òóọỏõôồốộổỗơờớợởỡ"
    r"ùúụủũưừứựửữ"
    r"ỳýỵỷỹđ"
    r"ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ"
    r"ÈÉẸẺẼÊỀẾỆỂỄ"
    r"ÌÍỊỈĨ"
    r"ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ"
    r"ÙÚỤỦŨƯỪỨỰỬỮ"
    r"ỲÝỴỶỸĐ]"
)

_STOP_SUFFIXES = [
    " tet", " tết", " vietnam", " viet nam", " việt nam", " traditional", " culture"
]


def normalize_spaces(text: str) -> str:
    return " ".join((text or "").strip().split())


def has_vietnamese_diacritic(text: str) -> bool:
    return bool(_VIETNAMESE_DIACRITIC_RE.search(text or ""))


def remove_vietnamese_diacritics(text: str) -> str:
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def make_keyword_slug(keyword: str) -> str:
    text = remove_vietnamese_diacritics(keyword).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def canonicalize_lookup_key(keyword: str) -> str:
    """Normalize a raw keyword for dictionary lookup."""
    key = normalize_spaces(keyword).lower()
    key = remove_vietnamese_diacritics(key)
    key = re.sub(r"[^a-z0-9\s]+", " ", key)
    key = normalize_spaces(key)

    for suffix in _STOP_SUFFIXES:
        if key.endswith(suffix.strip()):
            key = key[: -len(suffix.strip())].strip()

    return key


def load_keyword_mapping(path: Path) -> dict[str, str]:
    """
    Load mapping YAML.

    Supported format:
      keyword_normalization:
        "banh chung Tet": "bánh chưng"
        "ao dai": "áo dài"

    The returned dict contains both raw lower-case keys and canonicalized keys.
    """
    if not path.exists():
        raise FileNotFoundError(f"Keyword mapping file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw_map = data.get("keyword_normalization", data)
    if not isinstance(raw_map, dict):
        raise ValueError("Keyword mapping must be a dictionary.")

    mapping: dict[str, str] = {}
    for raw, normalized in raw_map.items():
        raw_text = normalize_spaces(str(raw))
        normalized_text = normalize_spaces(str(normalized))
        if not raw_text or not normalized_text:
            continue
        mapping[raw_text.lower()] = normalized_text
        mapping[canonicalize_lookup_key(raw_text)] = normalized_text

    return mapping


def normalize_keyword(raw_keyword: str, mapping: dict[str, str]) -> tuple[str, bool]:
    """
    Return (normalized_keyword, changed).
    Falls back to raw keyword when no mapping exists.
    """
    raw = normalize_spaces(raw_keyword)
    if not raw:
        return raw, False

    direct_key = raw.lower()
    canonical_key = canonicalize_lookup_key(raw)

    if direct_key in mapping:
        normalized = mapping[direct_key]
    elif canonical_key in mapping:
        normalized = mapping[canonical_key]
    else:
        normalized = raw

    return normalized, normalized != raw


def replace_keyword_in_text(text: str, raw_keyword: str, normalized_keyword: str) -> str:
    """Replace raw keyword occurrences in text, case-insensitive."""
    text = text or ""
    raw = normalize_spaces(raw_keyword)
    normalized = normalize_spaces(normalized_keyword)
    if not text or not raw or raw == normalized:
        return text

    # Replace exact raw keyword phrase, ignoring case.
    pattern = re.compile(re.escape(raw), flags=re.IGNORECASE)
    fixed = pattern.sub(normalized, text)

    # Also replace a canonical variant without common suffixes when it appears as ASCII text.
    canonical_raw = canonicalize_lookup_key(raw)
    if canonical_raw and canonical_raw != raw.lower():
        pattern2 = re.compile(re.escape(canonical_raw), flags=re.IGNORECASE)
        fixed = pattern2.sub(normalized, fixed)

    # Capitalize first character if needed for sentence-like standalone questions.
    if fixed:
        fixed = fixed[0].upper() + fixed[1:]
    return fixed


def normalize_processed_record(record: dict[str, Any], mapping: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize one final_train/val/test-style VQA record."""
    raw_keyword = str(record.get("keyword_raw") or record.get("keyword") or "").strip()
    normalized_keyword, changed = normalize_keyword(raw_keyword, mapping)
    keyword_slug = make_keyword_slug(normalized_keyword)

    fixed = dict(record)
    fixed["keyword_raw"] = raw_keyword
    fixed["keyword"] = normalized_keyword
    fixed["keyword_slug"] = keyword_slug

    old_standalone = str(record.get("standalone_question", ""))
    fixed["standalone_question"] = replace_keyword_in_text(
        old_standalone,
        raw_keyword=raw_keyword,
        normalized_keyword=normalized_keyword,
    )

    report = {
        "image_id": record.get("image_id"),
        "category": record.get("category"),
        "keyword_raw": raw_keyword,
        "keyword": normalized_keyword,
        "keyword_slug": keyword_slug,
        "changed": changed,
        "standalone_before": old_standalone,
        "standalone_after": fixed["standalone_question"],
    }
    return fixed, report


def normalize_knowledge_doc(doc: dict[str, Any], mapping: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize one knowledge_base.jsonl document."""
    metadata = dict(doc.get("metadata") or {})
    raw_keyword = str(metadata.get("keyword_raw") or metadata.get("keyword") or "").strip()
    normalized_keyword, changed = normalize_keyword(raw_keyword, mapping)
    keyword_slug = make_keyword_slug(normalized_keyword)

    fixed = dict(doc)
    fixed["schema_version"] = str(fixed.get("schema_version") or "1.0")

    metadata["keyword_raw"] = raw_keyword
    metadata["keyword"] = normalized_keyword
    metadata["keyword_slug"] = keyword_slug

    category = str(metadata.get("category") or "").strip()
    subcategory = str(metadata.get("subcategory") or keyword_slug).strip() or keyword_slug
    metadata["subcategory"] = subcategory

    old_content = str(doc.get("content", ""))
    cultural_context = old_content
    # Keep the original context but force the retrieval subject header to be normalized.
    if "Ngữ cảnh văn hóa:" in old_content:
        cultural_context = old_content.split("Ngữ cảnh văn hóa:", 1)[1].strip()

    fixed["content"] = (
        f"Chủ đề: {normalized_keyword}\n"
        f"Danh mục: {category}\n"
        f"Ngữ cảnh văn hóa: {cultural_context}"
    ).strip()
    fixed["metadata"] = metadata

    # Rebuild doc_id only when the existing doc_id clearly follows category|subcategory|hash.
    old_doc_id = str(doc.get("doc_id", "")).strip()
    parts = old_doc_id.split("|")
    if len(parts) >= 3 and category:
        fixed["doc_id"] = f"{category}|{keyword_slug}|{parts[-1]}"

    report = {
        "doc_id_before": old_doc_id,
        "doc_id_after": fixed.get("doc_id"),
        "category": category,
        "keyword_raw": raw_keyword,
        "keyword": normalized_keyword,
        "keyword_slug": keyword_slug,
        "changed": changed,
    }
    return fixed, report

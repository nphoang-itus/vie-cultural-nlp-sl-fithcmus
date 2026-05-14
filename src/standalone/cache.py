"""
Cache utilities for LLM standalone question rewriting.

Purpose:
- Avoid repeated Gemini calls for the same vision_caption + question.
- Preserve progress if the script crashes midway.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def normalize_for_cache(text: str) -> str:
    """Normalize text before hashing."""
    return " ".join((text or "").strip().split())


def make_rewrite_cache_key(vision_caption: str, question: str) -> str:
    """Create stable cache key from vision_caption + question."""
    raw = f"{normalize_for_cache(vision_caption)}||{normalize_for_cache(question)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_cache(path: Path) -> dict[str, Any]:
    """Load cache JSON. Return empty dict if file does not exist."""
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache: dict[str, Any], path: Path) -> None:
    """Save cache JSON atomically enough for local scripts."""
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    temp_path.replace(path)
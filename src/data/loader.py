"""
Raw data loading and schema validation.
Knows nothing about rewriting or retrieval — only reads and validates rows.
"""

from pathlib import Path
from typing import TypedDict, Iterator
from collections import defaultdict
from typing import Any
import json
import re


class VQARecord(TypedDict):
    """The schema for one processed VQA record. All fields are required."""

    # Unique ID for each image-question pair, e.g. "banh_chung_001_q1"
    image_id: str

    # Original image ID before adding question suffix
    base_image_id: str

    # Question index within the same image, e.g. "q1", "q2"
    question_id: str

    image_path: str
    category: str
    keyword: str
    
    vision_caption: str

    question: str
    standalone_question: str

    answer: str
    detailed_explanation: str
    cultural_context: str

    rewrite_method: str


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield one dict per line from a JSONL file. Skips blank lines."""
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            
            if not line:
                continue
            
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num} in {path}: {e}")


def load_raw_records(path: Path) -> list[dict]:
    """Load all records from a JSONL file into memory."""
    return list(iter_jsonl(path))

_IMAGE_QUESTION_ID_PATTERN = re.compile(r"^(?P<base_id>.+)_q(?P<question_number>\d+)$")

def validate_image_question_ids(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Validate image-question IDs from prepared raw records.

    Expected image_id format:
        <base_image_id>_q<number>

    Example:
        am_thuc|banh_chung|001_q1

    This function:
    - does not generate new image_id values
    - does not append new suffixes
    - fills base_image_id and question_id if they are missing
    - raises ValueError if image_id format is invalid
    """
    validated_records: list[dict[str, Any]] = []

    seen_image_ids: set[str] = set()

    for row_idx, record in enumerate(records):
        image_id = str(record.get("image_id", "")).strip()

        if not image_id:
            raise ValueError(f"Missing image_id at row {row_idx}")

        match = _IMAGE_QUESTION_ID_PATTERN.match(image_id)

        if not match:
            raise ValueError(
                f"Invalid image_id format at row {row_idx}: {image_id!r}. "
                "Expected format: <base_image_id>_q<number>, e.g. banh_chung_001_q1"
            )

        if image_id in seen_image_ids:
            raise ValueError(f"Duplicate image_id at row {row_idx}: {image_id!r}")

        seen_image_ids.add(image_id)

        base_image_id = record.get("base_image_id") or match.group("base_id")
        question_id = record.get("question_id") or f"q{match.group('question_number')}"

        validated_records.append({
            **record,
            "image_id": image_id,
            "base_image_id": str(base_image_id).strip(),
            "question_id": str(question_id).strip(),
        })

    return validated_records
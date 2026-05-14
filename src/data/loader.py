"""
Raw data loading and schema validation.
Knows nothing about rewriting or retrieval — only reads and validates rows.
"""

from pathlib import Path
from typing import TypedDict, Iterator
from collections import defaultdict
from typing import Any
import json


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

def normalize_image_question_ids(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Ensure each image-question pair has a unique image_id.

    Example:
        image_id = "banh_chung_001"
        question #1 -> "banh_chung_001_q1"
        question #2 -> "banh_chung_001_q2"

    This function does not mutate the input records.
    """
    counters: dict[str, int] = defaultdict(int)
    normalized_records: list[dict[str, Any]] = []

    for record in records:
        original_image_id = str(record.get("image_id", "")).strip()

        if not original_image_id:
            raise ValueError(f"Missing image_id in record: {record}")

        counters[original_image_id] += 1
        question_index = counters[original_image_id]

        question_id = f"q{question_index}"
        unique_image_id = f"{original_image_id}_{question_id}"

        normalized_records.append({
            **record,
            "base_image_id": original_image_id,
            "question_id": question_id,
            "image_id": unique_image_id,
        })

    return normalized_records
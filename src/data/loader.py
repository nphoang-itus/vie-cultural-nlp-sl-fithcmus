"""
Raw data loading and schema validation.
Knows nothing about rewriting or retrieval — only reads and validates rows.
"""

from pathlib import Path
from typing import TypedDict, Iterator
import json


class VQARecord(TypedDict):
    """The schema for one processed VQA record. All fields are required."""
    image_id: str
    image_path: str
    category: str
    keyword: str
    question: str
    standalone_question: str   # filled by standalone service; empty string until then
    answer: str
    detailed_explanation: str
    cultural_context: str
    rewrite_method: str        # "template" | "llm" | "unchanged" | "failed" | ""


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
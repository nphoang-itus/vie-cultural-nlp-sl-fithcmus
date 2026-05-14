"""
File I/O utilities. Writes JSONL records and JSON stats files.
No transformation logic lives here.
"""

import json
from pathlib import Path
from typing import Iterable


def write_jsonl(records: Iterable[dict], path: Path) -> int:
    """
    Write records to a JSONL file. Creates parent dirs if needed.
    Returns the number of records written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    count = 0
    with open(tmp_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    tmp_path.replace(path)
    return count


def write_json(data: dict, path: Path) -> None:
    """Write a dict as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)

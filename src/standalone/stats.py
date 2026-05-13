"""
Compute statistics over processed records. Pure computation — no I/O.
"""

from typing import Any

def compute_stats(
    original_records: list[dict[str, Any]],
    processed_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a stats dict suitable for writing to JSON."""
    total = len(processed_records)
    if total == 0:
        return {"total_questions": 0}

    methods = [r.get("rewrite_method", "") for r in processed_records]
    template_count = methods.count("template")
    llm_count      = methods.count("llm")
    unchanged_count = methods.count("unchanged")
    failed_count   = methods.count("failed")

    orig_lengths = [len(r.get("question", "")) for r in original_records]
    new_lengths  = [len(r.get("standalone_question", "")) for r in processed_records]

    examples = [
        {
            "keyword": r.get("keyword"),
            "original": r.get("question"),
            "standalone": r.get("standalone_question"),
            "method": r.get("rewrite_method"),
        }
        for r in processed_records[:5]   # first 5 as examples
    ]

    return {
        "total_questions": total,
        "template_rewritten_count": template_count,
        "llm_rewritten_count": llm_count,
        "unchanged_count": unchanged_count,
        "failed_count": failed_count,
        "template_rewrite_rate": round(template_count / total, 4),
        "llm_rewrite_rate": round(llm_count / total, 4),
        "average_original_question_length": round(sum(orig_lengths) / total, 2),
        "average_standalone_question_length": round(sum(new_lengths) / total, 2),
        "examples_before_after": examples,
    }